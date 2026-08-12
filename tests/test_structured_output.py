"""
Regression tests for provider-agnostic structured LLM extraction.

Mocks the chat model so tests remain deterministic and do not require
live OpenRouter/OpenAI calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from app.agent.nodes.requirement_collector import (
    RequirementExtractionError,
    collect_requirements,
)
from app.llm.client import StructuredOutputError, invoke_structured
from app.models.schemas import (
    InstanceRecommendation,
    TrafficPattern,
    TechnicalNeeds,
    UserRequirements,
    WorkloadType,
)


def _state(message: str):
    return {
        "requirements": UserRequirements(),
        "latest_user_message": message,
        "next_question": None,
        "technical_needs": None,
        "instance_candidates": None,
        "recommendation": None,
    }


@patch("app.llm.client.get_chat_model")
def test_invoke_structured_parses_valid_json_content(mock_get_model):
    model = MagicMock()
    model.bind.return_value = model
    model.invoke.return_value = AIMessage(
        content='{"workload_type":"web_app","registered_users":1000,"traffic_pattern":"steady"}'
    )
    model.with_structured_output.return_value.invoke.side_effect = AssertionError("fallback used")
    mock_get_model.return_value = model

    result = invoke_structured(UserRequirements, "extract requirements")
    assert result.workload_type == WorkloadType.WEB_APP
    assert result.registered_users == 1000
    assert result.traffic_pattern == TrafficPattern.STEADY


@patch("app.llm.client.get_chat_model")
def test_invoke_structured_accepts_fenced_json(mock_get_model):
    model = MagicMock()
    model.bind.return_value = model
    model.invoke.return_value = AIMessage(
        content='```json\n{"workload_type":"api_service","traffic_pattern":"bursty"}\n```'
    )
    model.with_structured_output.return_value.invoke.side_effect = AssertionError("fallback used")
    mock_get_model.return_value = model

    result = invoke_structured(UserRequirements, "extract")
    assert result.workload_type == WorkloadType.API_SERVICE
    assert result.traffic_pattern == TrafficPattern.BURSTY


@patch("app.llm.client.get_chat_model")
def test_invoke_structured_empty_response_raises_domain_error(mock_get_model):
    model = MagicMock()
    model.bind.return_value = model
    model.invoke.return_value = AIMessage(content="")
    model.with_structured_output.return_value.invoke.side_effect = TypeError(
        "'NoneType' object is not iterable"
    )
    mock_get_model.return_value = model

    with pytest.raises(StructuredOutputError, match="UserRequirements"):
        invoke_structured(UserRequirements, "extract")


@patch("app.llm.client.get_chat_model")
def test_invoke_structured_malformed_json_raises_domain_error(mock_get_model):
    model = MagicMock()
    model.bind.return_value = model
    model.invoke.return_value = AIMessage(content="sorry, I cannot help with that")
    model.with_structured_output.return_value.invoke.side_effect = TypeError(
        "'NoneType' object is not iterable"
    )
    mock_get_model.return_value = model

    with pytest.raises(StructuredOutputError, match="UserRequirements"):
        invoke_structured(UserRequirements, "extract")


@patch("app.llm.client.get_chat_model")
def test_invoke_structured_provider_choices_none_converted_to_domain_error(mock_get_model):
    """
    Simulates the live OpenRouter failure mode where structured-output
    parsing crashes because choices is None.
    """
    model = MagicMock()
    model.bind.return_value = model
    model.invoke.side_effect = TypeError("'NoneType' object is not iterable")
    model.with_structured_output.return_value.invoke.side_effect = TypeError(
        "'NoneType' object is not iterable"
    )
    mock_get_model.return_value = model

    with pytest.raises(StructuredOutputError) as exc_info:
        invoke_structured(UserRequirements, "extract")
    assert "NoneType" in str(exc_info.value) or "UserRequirements" in str(exc_info.value)


@patch("app.llm.client.get_chat_model")
def test_invoke_structured_falls_back_to_json_mode_when_text_unparsable(mock_get_model):
    model = MagicMock()
    model.bind.return_value = model
    model.invoke.return_value = AIMessage(content="not json")
    structured = MagicMock()
    structured.invoke.return_value = UserRequirements(
        workload_type=WorkloadType.WEB_APP,
        traffic_pattern=TrafficPattern.STEADY,
    )
    model.with_structured_output.return_value = structured
    mock_get_model.return_value = model

    result = invoke_structured(UserRequirements, "extract")
    assert result.workload_type == WorkloadType.WEB_APP
    model.with_structured_output.assert_called_once()
    assert model.with_structured_output.call_args.kwargs.get("method") == "json_mode"


@patch("app.agent.nodes.requirement_collector.invoke_structured")
def test_collect_requirements_uses_valid_structured_extraction(mock_invoke):
    mock_invoke.return_value = UserRequirements(
        workload_type=WorkloadType.WEB_APP,
        traffic_pattern=TrafficPattern.STEADY,
        registered_users=500,
    )
    state = collect_requirements(_state("It's a web application with 500 users, steady traffic."))
    assert state["requirements"].workload_type == WorkloadType.WEB_APP
    assert state["requirements"].registered_users == 500
    assert state["latest_user_message"] is None


@patch("app.agent.nodes.requirement_collector.invoke_structured")
def test_collect_requirements_converts_structured_failure_to_domain_error(mock_invoke):
    mock_invoke.side_effect = StructuredOutputError(
        "Failed to obtain valid UserRequirements from the LLM. "
        "JSON extraction error: 'NoneType' object is not iterable. "
        "Structured-output fallback error: 'NoneType' object is not iterable"
    )
    with pytest.raises(RequirementExtractionError, match="UserRequirements"):
        collect_requirements(_state("It's a web application."))


@patch("app.llm.client.get_chat_model")
def test_invoke_structured_unwraps_technical_needs_wrapper(mock_get_model):
    """
    Regression for a known provider behavior where the model returns:
      { "reasoning": "...", "technical_needs": { ...TechnicalNeeds... } }
    while the caller expects TechnicalNeeds directly.
    """
    model = MagicMock()
    model.bind.return_value = model
    model.invoke.return_value = AIMessage(
        content=(
            '{"reasoning":"wrapped ok",'
            '"technical_needs":{'
            '"estimated_concurrency":123,'
            '"resource_profile":"cpu_bound",'
            '"traffic_pattern":"bursty",'
            '"requires_gpu":false,'
            '"scaling_recommendation":"horizontal with auto scaling",'
            '"reasoning":"inner reasoning"'
            "}}"
        )
    )
    model.with_structured_output.return_value.invoke.side_effect = AssertionError(
        "fallback used"
    )
    mock_get_model.return_value = model

    result = invoke_structured(
        TechnicalNeeds, "Extract technical needs for this workload"
    )
    assert result.estimated_concurrency == 123
    assert result.resource_profile.value == "cpu_bound"
    assert result.traffic_pattern == TrafficPattern.BURSTY
    assert result.requires_gpu is False


@patch("app.llm.client.get_chat_model")
def test_invoke_structured_accepts_canonical_instance_recommendation_json(mock_get_model):
    model = MagicMock()
    model.bind.return_value = model
    model.invoke.return_value = AIMessage(
        content=(
            '{"recommended_instance":"m7i-flex.xlarge",'
            '"why":"Fits the workload",'
            '"assumptions":["steady traffic","low latency matters"],'
            '"confidence":"medium",'
            '"alternative_instance":"m8i-flex.xlarge",'
            '"trade_off":"More headroom for higher cost"}'
        )
    )
    model.with_structured_output.return_value.invoke.side_effect = AssertionError(
        "fallback used"
    )
    mock_get_model.return_value = model

    result = invoke_structured(InstanceRecommendation, "recommend")
    assert result.recommended_instance == "m7i-flex.xlarge"
    assert result.why == "Fits the workload"
    assert result.assumptions == ["steady traffic", "low latency matters"]
    assert result.confidence == "medium"
    assert result.alternative_instance == "m8i-flex.xlarge"
    assert result.trade_off == "More headroom for higher cost"


@patch("app.llm.client.get_chat_model")
def test_invoke_structured_normalizes_confidence_level_to_confidence(mock_get_model):
    model = MagicMock()
    model.bind.return_value = model
    model.invoke.return_value = AIMessage(
        content=(
            '{"recommended_instance":"m7i-flex.xlarge",'
            '"why":"Fits",'
            '"assumptions":["one assumption"],'
            '"confidence_level":"high",'
            '"alternative_instance":"m8i-flex.xlarge",'
            '"trade_off":"Better headroom"}'
        )
    )
    model.with_structured_output.return_value.invoke.side_effect = AssertionError(
        "fallback used"
    )
    mock_get_model.return_value = model

    result = invoke_structured(InstanceRecommendation, "recommend")
    assert result.confidence == "high"
    assert result.why == "Fits"


@patch("app.llm.client.get_chat_model")
def test_invoke_structured_normalizes_assumptions_string_to_list(mock_get_model):
    model = MagicMock()
    model.bind.return_value = model
    model.invoke.return_value = AIMessage(
        content=(
            '{"recommended_instance":"m7i-flex.xlarge",'
            '"why":"Fits",'
            '"assumptions":"one assumption only",'
            '"confidence":"medium"}'
        )
    )
    model.with_structured_output.return_value.invoke.side_effect = AssertionError(
        "fallback used"
    )
    mock_get_model.return_value = model

    result = invoke_structured(InstanceRecommendation, "recommend")
    assert result.assumptions == ["one assumption only"]


@patch("app.llm.client.get_chat_model")
def test_invoke_structured_normalizes_reasoning_to_why_for_instance_recommendation(
    mock_get_model,
):
    model = MagicMock()
    model.bind.return_value = model
    model.invoke.return_value = AIMessage(
        content=(
            '{"recommended_instance":"m7i-flex.xlarge",'
            '"reasoning":"This is the reason",'
            '"assumptions":["steady traffic"],'
            '"confidence":"low"}'
        )
    )
    model.with_structured_output.return_value.invoke.side_effect = AssertionError(
        "fallback used"
    )
    mock_get_model.return_value = model

    result = invoke_structured(InstanceRecommendation, "recommend")
    assert result.why == "This is the reason"
    assert result.confidence == "low"


@patch("app.llm.client.get_chat_model")
def test_invoke_structured_handles_combined_instance_recommendation_drift(mock_get_model):
    model = MagicMock()
    model.bind.return_value = model
    model.invoke.return_value = AIMessage(
        content=(
            '{"recommended_instance":"m7i-flex.xlarge",'
            '"alternative_instance":"m8i-flex.xlarge",'
            '"reasoning":"Fits the workload with steady low-latency needs",'
            '"assumptions":"single app tier",'
            '"confidence_level":"medium",'
            '"trade_off":"Higher cost for more headroom"}'
        )
    )
    model.with_structured_output.return_value.invoke.side_effect = AssertionError(
        "fallback used"
    )
    mock_get_model.return_value = model

    result = invoke_structured(InstanceRecommendation, "recommend")
    assert result.recommended_instance == "m7i-flex.xlarge"
    assert result.alternative_instance == "m8i-flex.xlarge"
    assert result.why == "Fits the workload with steady low-latency needs"
    assert result.assumptions == ["single app tier"]
    assert result.confidence == "medium"
    assert result.trade_off == "Higher cost for more headroom"


@patch("app.llm.client.get_chat_model")
def test_invoke_structured_missing_recommendation_data_fails_cleanly(mock_get_model):
    model = MagicMock()
    model.bind.return_value = model
    model.invoke.return_value = AIMessage(
        content=(
            '{"recommended_instance":"m7i-flex.xlarge",'
            '"assumptions":["steady traffic"]}'
        )
    )
    model.with_structured_output.return_value.invoke.side_effect = TypeError(
        "'NoneType' object is not iterable"
    )
    mock_get_model.return_value = model

    with pytest.raises(StructuredOutputError, match="InstanceRecommendation"):
        invoke_structured(InstanceRecommendation, "recommend")


@patch("app.llm.client.get_chat_model")
def test_invoke_structured_invalid_enum_fails_cleanly(mock_get_model):
    model = MagicMock()
    # resource_profile expects cpu_bound/memory_bound/...; provide an invalid value.
    model.invoke.return_value = AIMessage(
        content=(
            '{"estimated_concurrency":1,'
            '"resource_profile":"cpu_and_memory_bound",'
            '"traffic_pattern":"steady",'
            '"requires_gpu":false,'
            '"scaling_recommendation":"vertical",'
            '"reasoning":"bad enum"}'
        )
    )
    model.with_structured_output.return_value.invoke.side_effect = TypeError(
        "'NoneType' object is not iterable"
    )
    mock_get_model.return_value = model

    with pytest.raises(StructuredOutputError):
        invoke_structured(TechnicalNeeds, "Extract technical needs")

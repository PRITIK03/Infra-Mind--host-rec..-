# aws-instance-advisor

AWS Instance Advisor is an AI agent that gathers your workload requirements, reasons about the underlying technical needs (CPU, memory, traffic pattern), researches live AWS EC2 instance data, and recommends the optimal EC2 instance type along with the reasoning behind the choice.

## Running the project

- Prerequisite: Python 3.12+ in a virtual environment. The project is currently tested on Python 3.14.
- Install dependencies with `pip install -r requirements.txt`.
- Fill in `.env` from `.env.example`:
  - Required: `API_KEY`, `BASE_URL`, `MODEL_NAME` (LLM provider; currently OpenRouter), and `VANTAGE_API_KEY` (live EC2 specs).
  - Optional: `TAVILY_API_KEY` — enables one model-decided web-search round during system-design reasoning. If unset, the agent still runs using the LLM + live EC2 data.
- Run the CLI with `python -m app.main`.
- Run tests with `pytest -v`.
- V1 covers requirement collection plus EC2 instance recommendation using live AWS instance data and basic system design reasoning for horizontal vs. vertical scaling and GPU vs. CPU/memory-bound workloads.

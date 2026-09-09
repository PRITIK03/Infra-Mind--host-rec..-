"use client";

import { useEffect, useState } from "react";

interface Props {
  text: string;
}

// Characters revealed per tick — fast enough to feel snappy, slow
// enough to read as a deliberate reveal rather than a flash.
const CHARS_PER_TICK = 3;
const TICK_MS = 28;

export function ArchSummary({ text }: Props) {
  const [displayed, setDisplayed] = useState("");
  const [done, setDone] = useState(false);

  useEffect(() => {
    setDisplayed("");
    setDone(false);
    let idx = 0;

    const id = setInterval(() => {
      idx += CHARS_PER_TICK;
      if (idx >= text.length) {
        setDisplayed(text);
        setDone(true);
        clearInterval(id);
      } else {
        setDisplayed(text.slice(0, idx));
      }
    }, TICK_MS);

    return () => clearInterval(id);
  }, [text]);

  return (
    <p
      className={[
        "font-sans text-base leading-relaxed text-ink",
        !done ? "typewriter-cursor" : "",
      ].join(" ")}
    >
      {displayed}
    </p>
  );
}

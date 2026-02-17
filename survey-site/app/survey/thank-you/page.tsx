"use client";

import { useEffect, useMemo, useState } from "react";

type SubmitResponse = {
  ok: boolean;
};

type TextStore = {
  answers: Record<string, string | number | undefined>;
  attention_value: string;
};

type ImageStore = {
  answers: Record<string, string>;
  captcha_input: string;
  image_attention_choice: string;
};

async function logEvent(eventType: string, payload: Record<string, unknown>) {
  await fetch("/api/log", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ event_type: eventType, payload }),
  });
}

export default function ThankYouPage() {
  const [textData, setTextData] = useState<TextStore | null>(null);
  const [imageData, setImageData] = useState<ImageStore | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    const savedText = sessionStorage.getItem("survey_text_answers");
    const savedImage = sessionStorage.getItem("survey_image_answers");

    if (savedText) setTextData(JSON.parse(savedText) as TextStore);
    if (savedImage) setImageData(JSON.parse(savedImage) as ImageStore);

    void logEvent("thank_you_page_view", { page: "/survey/thank-you" });
  }, []);

  const ready = useMemo(() => Boolean(textData && imageData), [textData, imageData]);

  async function onSubmitAll() {
    if (!textData || !imageData) return;

    setSubmitting(true);
    try {
      const res = await fetch("/api/submit", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          text_answers: textData.answers,
          text_attention_value: textData.attention_value,
          image_answers: imageData.answers,
          captcha_input: imageData.captcha_input,
          image_attention_choice: imageData.image_attention_choice,
        }),
      });

      const json = (await res.json()) as SubmitResponse;
      await logEvent("final_submission_response", { ok: json.ok });

      if (json.ok) {
        sessionStorage.removeItem("survey_text_answers");
        sessionStorage.removeItem("survey_image_answers");
      }
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main
      style={{
        minHeight: "100vh",
        display: "grid",
        placeItems: "center",
        padding: 24,
      }}
    >
      <div style={{ width: "100%", maxWidth: 760 }}>
        <div
          style={{
            border: "1px solid #e2e2e2",
            borderRadius: 12,
            padding: 28,
            background: "#fafafa",
            textAlign: "center",
          }}
        >
          <h1 style={{ margin: 0, fontSize: 42, lineHeight: 1.2 }}>Thank You Page</h1>

          <div style={{ marginTop: 22 }}>
            <button
              onClick={onSubmitAll}
              disabled={!ready || submitting}
              style={{
                padding: "10px 18px",
                borderRadius: 8,
                border: "0",
                background: !ready || submitting ? "#98b3ed" : "#2c6bed",
                color: "white",
                cursor: !ready || submitting ? "default" : "pointer",
                fontWeight: 600,
              }}
            >
              {submitting ? "Submitting..." : "Submit Responses"}
            </button>
          </div>
        </div>
      </div>
    </main>
  );
}

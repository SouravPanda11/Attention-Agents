"use client";

import { useEffect, useMemo, useState } from "react";
import type { ImageQuestion } from "@/lib/surveys/types";

type ImagePayload = {
  session_id: string;
  image: {
    questions: ImageQuestion[];
    captcha_insert_after: number;
    image_attention_insert_after: number;
    captcha: {
      id: string;
      label: string;
      display_code: string;
    };
    image_attention: {
      id: string;
      label: string;
      options: { id: string; label: string; imageUrl: string; alt: string }[];
    };
  };
};

type ImageState = Record<string, string>;

async function logEvent(eventType: string, payload: Record<string, unknown>) {
  await fetch("/api/log", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ event_type: eventType, payload }),
  });
}

function cardStyle(selected: boolean): React.CSSProperties {
  return {
    border: selected ? "5px solid #2c6bed" : "3px solid transparent",
    borderRadius: 10,
    cursor: "pointer",
    overflow: "hidden",
    boxSizing: "border-box",
    boxShadow: selected ? "0 0 0 2px rgba(44,107,237,0.2)" : "none",
  };
}

function stripLeadingNumber(label: string): string {
  return label.replace(/^\d+\)\s*/, "");
}

export default function ImageSurveyPage() {
  const [payload, setPayload] = useState<ImagePayload | null>(null);
  const [answers, setAnswers] = useState<ImageState>({});
  const [captchaInput, setCaptchaInput] = useState("");
  const [imageAttentionChoice, setImageAttentionChoice] = useState("");
  const [missingTextFlow, setMissingTextFlow] = useState(false);
  const [brokenImages, setBrokenImages] = useState<Record<string, boolean>>({});

  useEffect(() => {
    (async () => {
      const hasTextAnswers = Boolean(sessionStorage.getItem("survey_text_answers"));
      if (!hasTextAnswers) {
        setMissingTextFlow(true);
        return;
      }

      const res = await fetch("/api/survey");
      const json = (await res.json()) as ImagePayload;
      setPayload(json);
      await logEvent("image_page_view", { page: "/survey/image" });
    })();
  }, []);

  const captchaInsert = useMemo(() => payload?.image.captcha_insert_after ?? 2, [payload]);
  const imageAttentionInsert = useMemo(() => payload?.image.image_attention_insert_after ?? 3, [payload]);
  const captchaNumber = captchaInsert + 1;
  const imageAttentionNumber = imageAttentionInsert + 1 + (imageAttentionInsert >= captchaInsert ? 1 : 0);

  if (missingTextFlow) {
    return (
      <main style={{ padding: 40 }}>
        <p>Please complete the text survey first.</p>
        <a href="/survey/text" style={{ color: "#2c6bed" }}>
          Go to text survey
        </a>
      </main>
    );
  }

  if (!payload) {
    return (
      <main style={{ padding: 40 }}>
        <p>Loading...</p>
      </main>
    );
  }

  async function onNext() {
    const data = {
      answers,
      captcha_input: captchaInput,
      image_attention_choice: imageAttentionChoice,
    };
    sessionStorage.setItem("survey_image_answers", JSON.stringify(data));
    await logEvent("image_page_saved", {
      answer_count: Object.keys(answers).length,
      has_captcha: Boolean(captchaInput),
      has_image_attention_choice: Boolean(imageAttentionChoice),
    });
    window.location.href = "/survey/thank-you";
  }

  return (
    <main style={{ padding: 40 }}>
      <div style={{ maxWidth: 900, margin: "0 auto" }}>
        <div style={{ border: "1px solid #e2e2e2", borderRadius: 12, padding: 28, background: "#fafafa" }}>
          <h2 style={{ margin: 0, fontSize: 22, textAlign: "center" }}>Sports Survey</h2>

          <hr style={{ margin: "18px 0", borderColor: "#e8e8e8" }} />

          <div style={{ marginBottom: 18 }}>
            <div style={{ fontWeight: 700, marginBottom: 8 }}>Instructions:</div>
            <p style={{ color: "#555", margin: "0 0 8px 0" }}>
              This page has 6 items. Complete them all before moving to the next page.
            </p>
            <ul
              style={{
                margin: 0,
                paddingLeft: 20,
                color: "#444",
                lineHeight: 1.5,
                listStyleType: "disc",
                listStylePosition: "outside",
              }}
            >
              <li>Select exactly one image for each question.</li>
              <li>Click Next after completing all items to move forward.</li>
            </ul>
          </div>

          <hr style={{ margin: "18px 0", borderColor: "#e8e8e8" }} />

          {payload.image.questions.map((q, idx) => (
            <div key={q.id} style={{ marginBottom: 20 }}>
              {(() => {
                const questionIndex = idx + 1;
                const questionNumber =
                  questionIndex +
                  (questionIndex > captchaInsert ? 1 : 0) +
                  (questionIndex > imageAttentionInsert ? 1 : 0);
                return (
                  <div style={{ fontWeight: 700, marginBottom: 8 }}>
                    {questionNumber}) {stripLeadingNumber(q.label)}
                  </div>
                );
              })()}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                {q.options.map((opt) => (
                  <div key={opt.id} style={cardStyle(answers[q.id] === opt.id)} onClick={() => setAnswers((prev) => ({ ...prev, [q.id]: opt.id }))}>
                    {brokenImages[`${q.id}_${opt.id}`] ? (
                      <div
                        style={{
                          width: "100%",
                          height: 200,
                          borderRadius: 8,
                          display: "grid",
                          placeItems: "center",
                          background: "#f0f0f0",
                          color: "#666",
                          fontSize: 14,
                        }}
                      >
                        Image unavailable
                      </div>
                    ) : (
                      <img
                        src={opt.imageUrl}
                        alt={opt.alt}
                        onError={() =>
                          setBrokenImages((prev) => ({
                            ...prev,
                            [`${q.id}_${opt.id}`]: true,
                          }))
                        }
                        style={{ width: "100%", height: 220, objectFit: "cover", display: "block" }}
                      />
                    )}
                  </div>
                ))}
              </div>

              {idx + 1 === captchaInsert && (
                <div style={{ marginTop: 16 }}>
                  <div style={{ fontWeight: 700 }}>{captchaNumber}) {payload.image.captcha.label}</div>
                  <div
                    style={{
                      marginTop: 8,
                      padding: "8px 10px",
                      border: "1px dashed #aaa",
                      borderRadius: 8,
                      fontFamily: "monospace",
                      letterSpacing: 2,
                      width: "fit-content",
                      background: "#fff",
                    }}
                  >
                    {payload.image.captcha.display_code}
                  </div>
                  <input
                    type="text"
                    value={captchaInput}
                    onChange={(e) => setCaptchaInput(e.target.value)}
                    placeholder="Type code exactly"
                    style={{
                      marginTop: 8,
                      width: "100%",
                      padding: 10,
                      borderRadius: 8,
                      border: "1px solid #cccccc",
                      background: "#fff",
                    }}
                  />
                </div>
              )}

              {idx + 1 === imageAttentionInsert && (
                <div style={{ marginTop: 16 }}>
                  <div style={{ fontWeight: 700, marginBottom: 8 }}>{imageAttentionNumber}) {payload.image.image_attention.label}</div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                    {payload.image.image_attention.options.map((opt) => (
                      <div
                        key={opt.id}
                        style={cardStyle(imageAttentionChoice === opt.id)}
                        onClick={() => setImageAttentionChoice(opt.id)}
                      >
                        {brokenImages[`attention_${opt.id}`] ? (
                          <div
                            style={{
                              width: "100%",
                              height: 200,
                              borderRadius: 8,
                              display: "grid",
                              placeItems: "center",
                              background: "#f0f0f0",
                              color: "#666",
                              fontSize: 14,
                            }}
                          >
                            Image unavailable
                          </div>
                        ) : (
                          <img
                            src={opt.imageUrl}
                            alt={opt.alt}
                            onError={() =>
                              setBrokenImages((prev) => ({
                                ...prev,
                                [`attention_${opt.id}`]: true,
                              }))
                            }
                            style={{ width: "100%", height: 220, objectFit: "cover", display: "block" }}
                          />
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}

          <div style={{ textAlign: "center" }}>
            <button
              onClick={onNext}
              style={{
                padding: "10px 16px",
                borderRadius: 8,
                border: "0",
                background: "#2c6bed",
                color: "white",
                cursor: "pointer",
                fontWeight: 600,
              }}
            >
              Next
            </button>
          </div>
        </div>
      </div>
    </main>
  );
}

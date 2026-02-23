"use client";

export default function DonePage() {
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
          <h1 style={{ margin: 0, fontSize: 36, lineHeight: 1.2 }}>Responses Submitted</h1>
          <p style={{ margin: "14px 0 0 0", color: "#555" }}>
            Your responses have been recorded successfully.
          </p>
        </div>
      </div>
    </main>
  );
}

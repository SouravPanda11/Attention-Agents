export default function WelcomePage() {
  return (
    <main
      style={{
        minHeight: "100vh",
        display: "grid",
        placeItems: "center",
        padding: 24,
      }}
    >
      <div style={{ textAlign: "center", maxWidth: 720 }}>
        <h1 style={{ margin: 0, fontSize: 28 }}>Survey Sandbox</h1>
        <p style={{ marginTop: 10, color: "#444", fontSize: 16 }}>
          This is a controlled survey website for Agentic-AI evaluation.
        </p>

        <div
          style={{
            display: "flex",
            gap: 16,
            justifyContent: "center",
            alignItems: "center",
            marginTop: 18,
            flexWrap: "wrap",
          }}
        >
          <a
            href="/survey/text"
            style={{
              textDecoration: "none",
              color: "#2c6bed",
              fontWeight: 600,
              fontSize: 16,
            }}
          >
            Start survey_v0 {"->"}
          </a>
          <a
            href="/survey_v1/text"
            style={{
              textDecoration: "none",
              color: "#2c6bed",
              fontWeight: 600,
              fontSize: 16,
            }}
          >
            Start survey_v1 {"->"}
          </a>
        </div>
      </div>
    </main>
  );
}

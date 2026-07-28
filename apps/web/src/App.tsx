function App() {
  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="ClearFrame home">
          <span className="brand-mark" aria-hidden="true">
            CF
          </span>
          <span>
            <strong>ClearFrame</strong>
            <small>Evidence redaction workspace</small>
          </span>
        </a>
        <span className="prototype-badge">Research prototype</span>
      </header>

      <section className="empty-state" aria-labelledby="welcome-title">
        <p className="eyebrow">Human review required</p>
        <h1 id="welcome-title">Redact sensitive video with confidence.</h1>
        <p>
          Upload, review AI-assisted proposals, and export an auditable derived copy while
          preserving the source video.
        </p>
        <div className="workflow" aria-label="Workflow">
          {["Upload", "Detect", "Review", "Export"].map((step, index) => (
            <div className="workflow-step" key={step}>
              <span>{String(index + 1).padStart(2, "0")}</span>
              {step}
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}

export default App;


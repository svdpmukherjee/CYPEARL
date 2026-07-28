import React, { useState } from "react";
import { rich } from "../content.jsx";

export default function Done({ content, completionCode, completionUrl }) {
  const t = content.done;
  const [copied, setCopied] = useState(false);

  const copyCode = async () => {
    try {
      await navigator.clipboard.writeText(completionCode);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (_) {
      // clipboard may be blocked (for example an insecure context); the code is
      // still visible and selectable, so the participant can copy it by hand
    }
  };

  return (
    <div className="card wide center">
      <h1>{t.title}</h1>
      <p className="lead">{rich(t.lead)}</p>

      {completionCode ? (
        <>
          <h2>{t.completionHeading}</h2>
          <p className="muted">{rich(t.completionHelp)}</p>
          <div className="ccodebox">
            <code className="ccode">{completionCode}</code>
            <button className="btn" onClick={copyCode}>
              {copied ? t.copiedLabel : t.copyButton}
            </button>
          </div>
          {completionUrl && (
            <div className="navbar center">
              <a className="btn primary" href={completionUrl} rel="noopener noreferrer">
                {t.submitButton}
              </a>
            </div>
          )}
        </>
      ) : (
        // no completion code configured (env var unset): show a plain fallback
        <p className="notice">{rich(t.completionMissing)}</p>
      )}
    </div>
  );
}

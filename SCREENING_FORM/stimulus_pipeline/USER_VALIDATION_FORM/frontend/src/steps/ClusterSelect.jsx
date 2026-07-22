import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { rich, clusterLabel } from "../content.jsx";

export default function ClusterSelect({ content, selected, onNext }) {
  const t = content.clusterSelect;
  const [clusters, setClusters] = useState([]);
  const [pick, setPick] = useState(selected);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState("");

  useEffect(() => {
    api
      .clusters()
      .then(setClusters)
      .catch((e) => setErr(e.message))
      .finally(() => setLoading(false));
  }, []);

  const chosen = clusters.find((c) => c.cluster === pick);

  return (
    <div className="card wide">
      <h1>{t.title}</h1>
      <p className="lead">{rich(t.lead)}</p>

      <h2>{t.selectPrompt}</h2>
      <p className="muted">{rich(t.selectHelp)}</p>

      {loading && <p className="muted">{t.loading}</p>}
      {err && <p className="error">{err}</p>}

      <div className="clustergrid">
        {clusters.map((c) => (
          <button
            key={c.cluster}
            className={"clustercard" + (pick === c.cluster ? " on" : "")}
            onClick={() => setPick(c.cluster)}
            type="button"
          >
            {/* Show only the job area. The specific role assigned to the
                participant is intentionally NOT revealed on this page. */}
            <span className="cname">{clusterLabel(content, c.cluster)}</span>
          </button>
        ))}
      </div>

      <div className="navbar">
        <span />
        <button
          className="btn primary"
          disabled={!chosen}
          onClick={() => onNext(chosen.cluster, chosen.recipientRole, chosen.note)}
        >
          {t.continueButton}
        </button>
      </div>
    </div>
  );
}

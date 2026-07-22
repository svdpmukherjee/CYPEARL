import React, { useMemo, useState } from "react";
import { rich, fmt } from "../content.jsx";

export default function EmailPage({
  content,
  email,
  index,
  total,
  participantName,
  saved,
  onBack,
  onNext,
}) {
  const t = content.email;
  // Q2 options: what may have felt off. Selecting any of these turns on the
  // per-section Edit buttons in the email. Both the keys (stored in the data)
  // and the labels are editable in content.json.
  const OFF_OPTIONS = t.offOptions.map((o) => [o.key, o.label]);
  const REALISM_LABELS = t.realismLabels;
  const [realism, setRealism] = useState(saved?.realism ?? null);
  const [reason, setReason] = useState(saved?.realismReason ?? "");
  const [offItems, setOffItems] = useState(saved?.offItems ?? []);
  const [edits, setEdits] = useState(saved?.sectionEdits ?? {});
  const [comment, setComment] = useState(saved?.comment ?? "");
  const [openEdits, setOpenEdits] = useState([]); // section ids currently editable
  const [tried, setTried] = useState(false);

  const editingEnabled = offItems.length > 0;

  // The email, broken into editable sections. Current value = edited value if
  // present, otherwise the original from the seeded email.
  const sections = useMemo(
    () => [
      {
        id: "subject",
        label: "Subject",
        type: "text",
        original: email.subject,
      },
      { id: "s1", label: "Opening", type: "area", original: email.body[0] },
      { id: "s2", label: "The request", type: "area", original: email.body[1] },
      { id: "link", label: "The link", type: "link", original: email.link },
      { id: "s3", label: "The urgency", type: "area", original: email.body[2] },
      {
        id: "s4",
        label: "The tone / framing",
        type: "area",
        original: email.body[3],
      },
      { id: "s5", label: "The closing", type: "area", original: email.body[4] },
    ],
    [email],
  );

  const valueOf = (sec) =>
    edits[sec.id] !== undefined ? edits[sec.id] : sec.original;
  const isEdited = (sec) => edits[sec.id] !== undefined;
  const isOpen = (id) => openEdits.includes(id);

  const toggleOff = (key) => {
    setOffItems((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key],
    );
  };

  const openEdit = (id) =>
    setOpenEdits((p) => (p.includes(id) ? p : [...p, id]));
  const closeEdit = (id) => setOpenEdits((p) => p.filter((x) => x !== id));

  const setEdit = (id, value) => setEdits((prev) => ({ ...prev, [id]: value }));
  const resetEdit = (id) =>
    setEdits((prev) => {
      const next = { ...prev };
      delete next[id];
      return next;
    });

  const c = email.conditions;

  // A one-line plain-language summary of this email's conditions, built from the
  // three parts in content.json and chosen by the email's own dir/urg/frame
  // codes. Falls back to the stored labels if a fragment is missing.
  const cond = t.conditions || {};
  const dirFrag = cond.dir?.[c.dir] || c.dir_label;
  const urgFrag = cond.urg?.[c.urg] || c.urg_label;
  const frameFrag = cond.frame?.[c.frame] || c.frame_label;
  const condSentence = `${dirFrag}, ${urgFrag}, ${frameFrag}.`;

  const reasonOk = reason.trim().length > 0;

  const next = () => {
    if (realism == null || !reasonOk) {
      setTried(true);
      return;
    }
    onNext({
      realism,
      realismReason: reason.trim(),
      offItems,
      sectionEdits: edits,
      comment,
    });
  };

  // --- render one email section (with an Edit button when editing is on) ---
  const renderSection = (sec) => {
    const open = isOpen(sec.id);
    const edited = isEdited(sec);

    let display;
    if (sec.type === "link") {
      const v = valueOf(sec);
      display = (
        <a className="mlink" href="#" onClick={(e) => e.preventDefault()}>
          {v.text}
        </a>
      );
    } else if (sec.type === "text") {
      display = (
        <div className="msubject">
          <span className="mfieldkey">{t.subjectLabel} </span>
          {valueOf(sec)}
        </div>
      );
    } else {
      display = <p className="mline">{valueOf(sec)}</p>;
    }

    return (
      <div
        className={
          "secrow" +
          (edited ? " edited" : "") +
          (sec.id === "subject" ? " subjectrow" : "")
        }
        key={sec.id}
      >
        <div className="seccontent">
          {open ? (
            sec.type === "link" ? (
              <div className="editfields">
                <label>{t.linkTextLabel}</label>
                <input
                  type="text"
                  value={valueOf(sec).text}
                  onChange={(e) =>
                    setEdit(sec.id, { ...valueOf(sec), text: e.target.value })
                  }
                />
                <label>{t.linkUrlLabel}</label>
                <input
                  type="text"
                  value={valueOf(sec).url}
                  onChange={(e) =>
                    setEdit(sec.id, { ...valueOf(sec), url: e.target.value })
                  }
                />
              </div>
            ) : (
              <>
                {sec.id === "subject" && (
                  <div className="mfieldkey">{t.subjectLabel}</div>
                )}
                <textarea
                  className="editarea"
                  value={valueOf(sec)}
                  rows={sec.type === "text" ? 1 : 2}
                  onChange={(e) => setEdit(sec.id, e.target.value)}
                />
              </>
            )
          ) : (
            display
          )}
        </div>

        {editingEnabled && (
          <div className="secactions">
            {open ? (
              <>
                <button
                  className="editbtn done"
                  onClick={() => closeEdit(sec.id)}
                >
                  {t.doneButton}
                </button>
                {edited && (
                  <button
                    className="editbtn reset"
                    onClick={() => {
                      resetEdit(sec.id);
                      closeEdit(sec.id);
                    }}
                  >
                    {t.resetButton}
                  </button>
                )}
              </>
            ) : (
              <button className="editbtn" onClick={() => openEdit(sec.id)}>
                {edited ? t.editAgainButton : t.editButton}
              </button>
            )}
          </div>
        )}
      </div>
    );
  };

  const bySide = (id) => sections.find((s) => s.id === id);

  return (
    <div className="emailpage">
      <div className="emailhead">
        <span className="idx">
          Email {index + 1} of {total}
        </span>
      </div>

      {/* the email, single black colour, full signature */}
      <div className="mailcard">
        {/* conditions summary sits at the top of the email itself */}
        <div className="condbanner">
          <div className="condtitle">{t.conditionsTitle}</div>
          <p className="condsentence">{rich(condSentence)}</p>
        </div>
        {renderSection(bySide("subject"))}
        <div className="mgreet">
          {fmt(t.greeting, { name: participantName || "there" })}
        </div>
        {renderSection(bySide("s1"))}
        {renderSection(bySide("s2"))}
        {renderSection(bySide("link"))}
        {renderSection(bySide("s3"))}
        {renderSection(bySide("s4"))}
        {renderSection(bySide("s5"))}

        <div className="msign">
          <div>{t.signOff}</div>
          <div className="signname">{email.sender.name}</div>
          <div>{email.sender.title}</div>
          <div className="signcontact">{email.sender.contact}</div>
          <div className="signcontact">
            {t.mobileLabel} {email.sender.mobile}
          </div>
        </div>
      </div>

      {/* all three questions in one panel */}
      <div className="questions">
        <div className="questionsintro">{rich(t.questionsIntro)}</div>

        {/* Q1 realism */}
        <div className="qblock">
          <div className="qh">{t.q1Heading}</div>
          <div className="qsub">
            {rich(t.q1Sub, { role: email.recipient_role })}
          </div>
          <div className="scale">
            {[1, 2, 3, 4, 5].map((v) => (
              <button
                key={v}
                type="button"
                className={"scaleopt" + (realism === v ? " on" : "")}
                onClick={() => setRealism(v)}
              >
                <span className="n">{v}</span>
                <span className="t">{REALISM_LABELS[v - 1]}</span>
              </button>
            ))}
          </div>
          {tried && realism == null && (
            <div className="error small">{t.q1Required}</div>
          )}

          <div className="reasonwrap">
            <div className="qh small">{t.reasonHeading}</div>
            <div className="qsub">{rich(t.reasonSub, { role: email.recipient_role })}</div>
            <textarea
              className="commentarea"
              placeholder={t.reasonPlaceholder}
              value={reason}
              onChange={(e) => setReason(e.target.value)}
            />
            {tried && !reasonOk && (
              <div className="error small">{t.reasonRequired}</div>
            )}
          </div>
        </div>

        {/* Q2 off flags */}
        <div className="qblock">
          <div className="qh">
            {t.q2Heading} <span className="muted small">{t.q2Optional}</span>
          </div>
          <div className="qsub">{rich(t.q2Sub)}</div>
          <div className="flags">
            {OFF_OPTIONS.map(([key, label]) => (
              <button
                key={key}
                type="button"
                className={"flag" + (offItems.includes(key) ? " on" : "")}
                onClick={() => toggleOff(key)}
              >
                {label}
              </button>
            ))}
          </div>

          {editingEnabled && (
            <div className="editcallout">{rich(t.editCallout)}</div>
          )}
        </div>

        {/* optional comment */}
        <div className="qblock">
          <div className="qh">{t.commentHeading}</div>
          <textarea
            className="commentarea"
            placeholder={t.commentPlaceholder}
            value={comment}
            onChange={(e) => setComment(e.target.value)}
          />
        </div>
      </div>

      <div className="navbar sticky">
        <button className="btn" onClick={onBack}>
          {t.backButton}
        </button>
        <button className="btn primary" onClick={next}>
          {index + 1 >= total ? t.finishButton : t.nextButton}
        </button>
      </div>
    </div>
  );
}

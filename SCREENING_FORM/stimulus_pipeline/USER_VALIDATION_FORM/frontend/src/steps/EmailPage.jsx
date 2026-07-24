import React, { useState } from "react";
import { rich, fmt } from "../content.jsx";
import RatingSlider from "../RatingSlider.jsx";

// One email page. The email stays visible at the top the whole time. Below it
// the participant answers three questions, ONE at a time, each replacing the
// previous one in the same place:
//   1. a realism rating (text labels, no numbers)
//   2. a short reason for that rating
//   3. would you change anything (optional note) with the option to edit the
//      email in place. Only the subject and the message body are editable; the
//      sender, the link, and the signature stay fixed so the manipulated sender
//      and the constant link are preserved.
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
  const REALISM_LABELS = t.realismLabels;

  const [subStep, setSubStep] = useState(1); // 1 = rating, 2 = reason, 3 = change
  const [realism, setRealism] = useState(saved?.realism ?? 5); // slider starts at the midpoint
  const [reason, setReason] = useState(saved?.realismReason ?? "");
  const [changeText, setChangeText] = useState(saved?.changeText ?? "");
  const [tried, setTried] = useState(false);

  // Working copy of the editable parts. Seeded from a previously saved edit so
  // returning to an email shows what the participant already wrote.
  const [editing, setEditing] = useState(false);
  const [editSubject, setEditSubject] = useState(
    saved?.editedEmail?.subject ?? email.subject
  );
  const [editBody, setEditBody] = useState(
    saved?.editedEmail?.body ?? [...email.body]
  );

  // A one-line plain-language summary of this email's conditions, worded to
  // match the eight situations on the "Before you begin reading" page.
  const c = email.conditions;
  const cond = t.conditions || {};
  const dirFrag = cond.dir?.[c.dir] || c.dir_label;
  const urgFrag = cond.urg?.[c.urg] || c.urg_label;
  const frameFrag = cond.frame?.[c.frame] || c.frame_label;

  const reasonOk = reason.trim().length > 0;

  const setBodyLine = (i, val) =>
    setEditBody((prev) => prev.map((b, j) => (j === i ? val : b)));

  const resetEdits = () => {
    setEditSubject(email.subject);
    setEditBody([...email.body]);
  };

  const back = () => {
    if (subStep === 1) return onBack();
    setTried(false);
    setSubStep(subStep - 1);
  };

  const forward = () => {
    if (subStep === 1) {
      if (realism == null) return setTried(true);
      setTried(false);
      return setSubStep(2);
    }
    if (subStep === 2) {
      if (!reasonOk) return setTried(true);
      setTried(false);
      return setSubStep(3);
    }
    // subStep 3: finish this email
    const changed =
      editSubject !== email.subject ||
      editBody.some((b, i) => b !== email.body[i]);
    onNext({
      realism,
      realismReason: reason.trim(),
      changeText: changeText.trim(),
      editedEmail: changed
        ? { subject: editSubject, body: [...editBody] }
        : null,
    });
  };

  // A body line: a static paragraph normally, or a textarea while editing.
  const bodyLine = (i) =>
    editing ? (
      <textarea
        key={i}
        className="editarea mline"
        value={editBody[i]}
        onChange={(e) => setBodyLine(i, e.target.value)}
        rows={2}
      />
    ) : (
      <p className="mline" key={i}>
        {editBody[i]}
      </p>
    );

  const primaryLabel =
    subStep < 3
      ? t.continueButton
      : index + 1 >= total
      ? t.finishButton
      : t.nextButton;

  return (
    <div className="emailpage">
      <div className="emailhead">
        <span className="idx">
          Email {index + 1} of {total}
        </span>
      </div>

      {/* the email: single ink, full signature, editable subject + body on Q3 */}
      <div className={"mailcard" + (editing ? " editing" : "")}>
        <div className="condbanner">
          <div className="condtitle">{t.conditionsTitle}</div>
          <p className="condsentence">
            {rich(t.conditionsSentence, {
              dir: dirFrag,
              urg: urgFrag,
              frame: frameFrag,
            })}
          </p>
          {t.conditionsLead && (
            <p className="condlead">{rich(t.conditionsLead)}</p>
          )}
        </div>

        <div className="secrow subjectrow">
          <div className="seccontent">
            <div className="msubject">
              <span className="mfieldkey">{t.subjectLabel} </span>
              {editing ? (
                <input
                  className="editarea subjectedit"
                  value={editSubject}
                  onChange={(e) => setEditSubject(e.target.value)}
                />
              ) : (
                editSubject
              )}
            </div>
          </div>
        </div>

        <div className="mgreet">
          {fmt(t.greeting, { name: participantName || "there" })}
        </div>

        {bodyLine(0)}
        {bodyLine(1)}
        <p className="mline">
          <a className="mlink" href="#" onClick={(e) => e.preventDefault()}>
            {email.link.text}
          </a>
        </p>
        {bodyLine(2)}
        {bodyLine(3)}
        {bodyLine(4)}

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

      {/* one question at a time, below the email */}
      <div className="questions">
        <div className="questionsintro">{rich(t.questionsIntro)}</div>
        <div className="substepmeta">{fmt(t.stepLabel, { n: subStep })}</div>

        {subStep === 1 && (
          <div className="qblock">
            <div className="qh">{t.q1Heading}</div>
            <div className="qsub">
              {rich(t.q1Sub, { role: email.recipient_role })}
            </div>
            <RatingSlider
              value={realism}
              onChange={(v) => {
                setRealism(v);
                setTried(false);
              }}
              min={1}
              max={10}
              minLabel={REALISM_LABELS[0]}
              maxLabel={REALISM_LABELS[REALISM_LABELS.length - 1]}
              ariaLabel="How realistic this email is for your role"
            />
            {tried && realism == null && (
              <div className="error small">{t.q1Required}</div>
            )}
          </div>
        )}

        {subStep === 2 && (
          <div className="qblock">
            <div className="qh">{t.q2Heading}</div>
            <div className="qsub">
              {rich(t.q2Sub, { role: email.recipient_role })}
            </div>
            <textarea
              className="commentarea"
              placeholder={t.q2Placeholder}
              value={reason}
              onChange={(e) => setReason(e.target.value)}
            />
            {tried && !reasonOk && (
              <div className="error small">{t.q2Required}</div>
            )}
          </div>
        )}

        {subStep === 3 && (
          <div className="qblock">
            <div className="qh">{t.q3Heading}</div>
            <div className="qsub">
              {rich(t.q3Sub, { role: email.recipient_role })}
            </div>

            <div className="edittoolbar">
              {!editing ? (
                <button
                  type="button"
                  className="editbtn"
                  onClick={() => setEditing(true)}
                >
                  {t.q3EditButton}
                </button>
              ) : (
                <>
                  <button
                    type="button"
                    className="editbtn done"
                    onClick={() => setEditing(false)}
                  >
                    {t.q3DoneButton}
                  </button>
                  <button
                    type="button"
                    className="editbtn reset"
                    onClick={resetEdits}
                  >
                    {t.q3ResetButton}
                  </button>
                </>
              )}
            </div>
            {editing && <div className="edithint">{rich(t.q3EditHint)}</div>}

            <div className="reasonwrap">
              <div className="qh small">{t.q3NoteHeading}</div>
              <textarea
                className="commentarea"
                placeholder={t.q3NotePlaceholder}
                value={changeText}
                onChange={(e) => setChangeText(e.target.value)}
              />
            </div>
          </div>
        )}
      </div>

      <div className="navbar sticky">
        <button className="btn" onClick={back}>
          {t.backButton}
        </button>
        <button className="btn primary" onClick={forward}>
          {primaryLabel}
        </button>
      </div>
    </div>
  );
}

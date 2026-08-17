import React, { useState } from "react";
import { rich, fmt } from "../content.jsx";
import RatingBoxes from "../RatingBoxes.jsx";

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
  // Starts unanswered. It used to default to the slider's midpoint (5), which
  // meant an untouched scale was stored as a real rating of 5 and the "please
  // choose a rating" check below could never fire, so a participant could click
  // through all 16 emails and record a 5 for every one.
  const [realism, setRealism] = useState(saved?.realism ?? null);
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

  // The one-line summary of this email's conditions. It is taken VERBATIM from
  // the matching situation on the first-task page, looked up by this email's
  // own combination, so the participant reads exactly the sentence they already
  // rated rather than a reworded version of it. That page is therefore the
  // single place this wording is edited.
  // The fallback assembles the sentence from email.conditions instead, and only
  // comes into play if a combination has no matching situation (for example if
  // one is deleted from priorJudgments.items).
  const c = email.conditions;
  const cond = t.conditions || {};
  const dirFrag = cond.dir?.[c.dir] || c.dir_label;
  const urgFrag = cond.urg?.[c.urg] || c.urg_label;
  const frameFrag = cond.frame?.[c.frame] || c.frame_label;
  const comboKey = `${c.dir}_${c.urg}_${c.frame}`;
  const situation =
    (content.priorJudgments?.items || []).find((it) => it.key === comboKey)
      ?.text ||
    fmt(t.conditionsSentence, {
      dir: dirFrag,
      urg: urgFrag,
      frame: frameFrag,
    });

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
    // subStep 3: finish this email. The working copy (editSubject / editBody) is
    // updated on every keystroke, so we save it here whether or not the
    // participant clicked "Done editing" first: leaving edit mode open and
    // pressing "Next email" still carries their edited version to the database.
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
        <div className={"condbanner" + (editing ? " editmode" : "")}>
          <div className="condtitle">
            {editing ? t.editBannerTitle : t.conditionsTitle}
          </div>
          <p className="condsentence">
            {editing
              ? rich(t.editBannerSentence, { situation })
              : rich(situation)}
          </p>
          {editing
            ? t.editBannerLead && (
                <p className="condlead">{rich(t.editBannerLead)}</p>
              )
            : t.conditionsLead && (
                <p className="condlead">{rich(t.conditionsLead)}</p>
              )}
          {editing && t.editBannerWarn && (
            <p className="condwarn">{rich(t.editBannerWarn)}</p>
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
            {/* Same control and same stored 1..10 scale as the eight scenarios
                on the "first task" page, so a participant's prior expectation
                and their rating of an actual email stay directly comparable.
                The name is per email, or all 16 radio groups would merge. */}
            <RatingBoxes
              name={"realism_" + email.src}
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
            <div className="qh">
              {rich(t.q3Heading, { role: email.recipient_role })}
            </div>
            <div className="qsub">
              {rich(t.q3Sub, { role: email.recipient_role })}
            </div>

            <div className="edittoolbar">
              {!editing ? (
                <button
                  type="button"
                  className="editbtn trigger"
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
            {editing && (
              <div className="edithint">
                {rich(t.q3EditHint, {
                  dir: dirFrag,
                  urg: urgFrag,
                  frame: frameFrag,
                })}
              </div>
            )}

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

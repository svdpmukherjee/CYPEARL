import React from "react";
import { rich } from "../content.jsx";

// Page 3 of 5: what the follow-up study would actually ask of them.
//
// ITS POSITION IS THE POINT. It sits BELOW the job areas (page 2) and ABOVE the
// per-area questions (page 4), and both halves of that matter:
//
//   below page 2, because somebody who reads this first learns what we are
//   recruiting for, and can claim whichever areas look most likely to get them
//   into a paid study rather than the ones they have actually worked in;
//
//   above page 4, because "do you know this kind of role well enough" is
//   unanswerable until you know what the judgment is, and the opt-in on page 5
//   is not informed unless the task was described before it.
//
// The copy is the main study's framing cut down to two paragraphs: what the
// judgment is, then the three things that go into it. It names no role, because
// none is assigned until the follow-up study. Rendering is the same `.objective`
// markup as the main study's landing page, so a participant who gets invited
// meets a longer version of a panel they have already seen.
export default function Task({ content, onNext }) {
  const explainer = (content.task && content.task.taskExplainer) || {};

  return (
    <div className="card wide">
      <div className="focusblock">
        <section className="objective">
          <div className="objectiveband">{explainer.eyebrow}</div>
          <div className="objectivebody">
            {explainer.heading && (
              <h3 className="objectiveheading">{rich(explainer.heading)}</h3>
            )}
            <p className="objectivelead">{rich(explainer.lead)}</p>

            {/* The three things they would weigh, in prose. They were three
                tiles until the labels turned out to be the problem: "the
                urgency" and "what is at stake" are our shorthand and read cold
                they do not say what we mean, so a sentence each that names the
                actual dimension beats a two-word heading. Still QUESTIONS the
                participant answers, never answers we supply: naming which
                combinations are realistic hands them a rule and they stop
                judging, which is the measurement. */}
            {explainer.detail && (
              <p className="objectivelead">{rich(explainer.detail)}</p>
            )}

            {/* Empty in the current copy: the lead names the wrong question and
                the right one in a single sentence, where the main study's paid
                briefing spends two labelled rows on it. Fill `contrast` back in
                and the rows render again. */}
            {Array.isArray(explainer.contrast) && explainer.contrast.length > 0 && (
              <div className="contrast">
                {explainer.contrast.map((row, i) => (
                  <div key={i} className="contrastrow">
                    <span className="contrastlabel">{row.label}</span>
                    <p className="contrasttext">{rich(row.text)}</p>
                  </div>
                ))}
              </div>
            )}

            {/* Empty in the current copy: the detail paragraph above says the
                same three things in prose. Fill `factors` back in and the tiles
                render again. */}
            {Array.isArray(explainer.factors) && explainer.factors.length > 0 && (
              <div className="weighgrid">
                {explainer.factors.map((f, i) => (
                  <div key={i} className="weighcard">
                    <span className="weighlabel">{f.label}</span>
                    <p className="weighq">{rich(f.text)}</p>
                  </div>
                ))}
              </div>
            )}

            {/* The one line that has to land. Its own callout, not a trailing
                paragraph: someone who has never held the exact job title reads
                "one work role" and counts themselves out, and they are exactly
                who we want, because the areas overlap and the assignment fills
                a scarce cluster from an abundant one. */}
            {explainer.closing && (
              <p className="objectivekeynote">{rich(explainer.closing)}</p>
            )}
          </div>
        </section>
      </div>

      <div className="navbar end">
        <button className="btn primary" onClick={onNext}>
          {content.task.continueButton}
        </button>
      </div>
    </div>
  );
}

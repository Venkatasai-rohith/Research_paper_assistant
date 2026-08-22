import type { Citation } from "../types";

interface CitationChipsProps {
  citations: Citation[];
}

export function CitationChips({ citations }: CitationChipsProps) {
  if (citations.length === 0) return null;

  const unique = citations.filter(
    (c, i, arr) =>
      arr.findIndex((x) => x.title === c.title && x.page === c.page) === i
  );

  return (
    <div className="citations">
      {unique.map((citation, i) => (
        <span key={i} className="citation-chip" title={citation.title}>
          <span className="citation-chip__page">p.{citation.page}</span>
          {citation.title.length > 30
            ? citation.title.slice(0, 30) + "…"
            : citation.title}
        </span>
      ))}
    </div>
  );
}

import type { Paper } from "../types";
import { UploadZone } from "./UploadZone";

interface PaperLibraryProps {
  papers: Paper[];
  selectedIds: Set<string>;
  onToggle: (id: string) => void;
  onSelectAll: () => void;
  onClearSelection: () => void;
  onDelete: (id: string) => void;
  onUpload: (files: File[]) => void;
  uploading: boolean;
}

export function PaperLibrary({
  papers,
  selectedIds,
  onToggle,
  onSelectAll,
  onClearSelection,
  onDelete,
  onUpload,
  uploading,
}: PaperLibraryProps) {
  return (
    <aside className="sidebar">
      <div className="sidebar__header">
        <div className="sidebar__title">Paper Library</div>
        <UploadZone onUpload={onUpload} uploading={uploading} />
        {papers.length > 0 && (
          <div className="sidebar__actions">
            <button className="btn btn--ghost btn--sm" onClick={onSelectAll}>
              Select all
            </button>
            {selectedIds.size > 0 && (
              <button className="btn btn--ghost btn--sm" onClick={onClearSelection}>
                Clear
              </button>
            )}
          </div>
        )}
      </div>

      {selectedIds.size > 0 && (
        <div className="selected-count">
          <strong>{selectedIds.size}</strong> paper{selectedIds.size !== 1 ? "s" : ""} selected
        </div>
      )}

      <div className="sidebar__list">
        {papers.length === 0 ? (
          <div className="sidebar__empty">
            No papers yet.
            <br />
            Upload a PDF to get started.
          </div>
        ) : (
          papers.map((paper) => (
            <div
              key={paper.paper_id}
              className={`paper-item${selectedIds.has(paper.paper_id) ? " paper-item--selected" : ""}`}
              onClick={() => onToggle(paper.paper_id)}
            >
              <input
                type="checkbox"
                className="paper-item__checkbox"
                checked={selectedIds.has(paper.paper_id)}
                onChange={() => onToggle(paper.paper_id)}
                onClick={(e) => e.stopPropagation()}
              />
              <div className="paper-item__content">
                <div className="paper-item__title" title={paper.title}>
                  {paper.title.replace(/\.pdf$/i, "")}
                </div>
                <div className="paper-item__id">{paper.paper_id.slice(0, 8)}…</div>
              </div>
              <button
                className="paper-item__delete"
                title="Delete paper"
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(paper.paper_id);
                }}
              >
                ✕
              </button>
            </div>
          ))
        )}
      </div>
    </aside>
  );
}

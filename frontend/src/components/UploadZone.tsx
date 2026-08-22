import { useRef, useState } from "react";

interface UploadZoneProps {
  onUpload: (files: File[]) => void;
  uploading: boolean;
}

export function UploadZone({ onUpload, uploading }: UploadZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);

  function handleFiles(fileList: FileList | null) {
    if (!fileList || fileList.length === 0) return;
    const pdfs = Array.from(fileList).filter(
      (f) => f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf")
    );
    if (pdfs.length > 0) onUpload(pdfs);
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragActive(false);
    handleFiles(e.dataTransfer.files);
  }

  return (
    <div
      className={`upload-zone${dragActive ? " upload-zone--active" : ""}`}
      onClick={() => !uploading && inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setDragActive(true);
      }}
      onDragLeave={() => setDragActive(false)}
      onDrop={handleDrop}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,application/pdf"
        multiple
        onChange={(e) => handleFiles(e.target.files)}
      />
      {uploading ? (
        <div className="upload-progress">
          <div className="spinner" />
          Indexing papers…
        </div>
      ) : (
        <>
          <div className="upload-zone__icon">📄</div>
          <div className="upload-zone__text">Drop PDFs here or click to upload</div>
          <div className="upload-zone__hint">Multiple files supported</div>
        </>
      )}
    </div>
  );
}

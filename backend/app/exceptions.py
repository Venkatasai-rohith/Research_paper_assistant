class PDFExtractionError(Exception):
    """Raised when PDF text extraction fails."""

class VectorStoreCreationError(Exception):
    """Raised when Vectorstore creation fails."""

class PaperNotFoundError(Exception):
    """Raised when paper loading fails."""
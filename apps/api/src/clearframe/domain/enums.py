from enum import StrEnum


class VideoStatus(StrEnum):
    UPLOADING = "UPLOADING"
    VALIDATING = "VALIDATING"
    PROXYING = "PROXYING"
    PROCESSING = "PROCESSING"
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    EXPORTING = "EXPORTING"
    EXPORTED = "EXPORTED"
    FAILED = "FAILED"


class RunStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TrackStatus(StrEnum):
    PROPOSED = "PROPOSED"
    UNCERTAIN = "UNCERTAIN"
    REVIEWED = "REVIEWED"


class TrackSource(StrEnum):
    MODEL = "MODEL"
    MANUAL = "MANUAL"
    REPROCESSING = "REPROCESSING"


class RedactionClass(StrEnum):
    FACE = "face"
    LICENSE_PLATE = "license_plate"
    SCENE_TEXT = "scene_text"


class ProposalSource(StrEnum):
    MODEL = "MODEL"
    MOCK = "MOCK"
    MANUAL = "MANUAL"
    REPROCESSING = "REPROCESSING"


class InterpolationMode(StrEnum):
    LINEAR = "LINEAR"
    STATIC = "STATIC"
    TRACKED = "TRACKED"


class ReviewActionType(StrEnum):
    ACCEPT_PROPOSAL = "ACCEPT_PROPOSAL"
    BULK_ACCEPT = "BULK_ACCEPT"
    RESTORE_TRACK = "RESTORE_TRACK"
    REDACT_TRACK = "REDACT_TRACK"
    DELETE_FALSE_POSITIVE = "DELETE_FALSE_POSITIVE"
    CREATE_MANUAL_REGION = "CREATE_MANUAL_REGION"
    RESIZE_REGION = "RESIZE_REGION"
    MOVE_REGION = "MOVE_REGION"
    SPLIT_TRACK = "SPLIT_TRACK"
    MERGE_TRACKS = "MERGE_TRACKS"
    EXTEND_TRACK = "EXTEND_TRACK"
    TRIM_TRACK = "TRIM_TRACK"
    CHANGE_CLASS = "CHANGE_CLASS"
    UNDO = "UNDO"
    REDO = "REDO"
    EXPORT_REQUESTED = "EXPORT_REQUESTED"
    EXPORT_COMPLETED = "EXPORT_COMPLETED"


class ExportStatus(StrEnum):
    QUEUED = "QUEUED"
    RENDERING = "RENDERING"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class RedactionStyle(StrEnum):
    PIXELATE = "pixelate"
    GAUSSIAN_BLUR = "gaussian_blur"
    BLACK_BOX = "black_box"


class RedactionShape(StrEnum):
    RECTANGLE = "rectangle"
    ELLIPSE = "ellipse"


class JobType(StrEnum):
    INGEST = "INGEST"
    PROXY = "PROXY"
    DETECT = "DETECT"
    REPROCESS = "REPROCESS"
    EXPORT = "EXPORT"


class JobStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ReprocessingSuggestionStatus(StrEnum):
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    DISMISSED = "DISMISSED"

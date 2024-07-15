ALGO_NAME: str = 'algoName'
ALGO_GROUP: str = 'algoGroup'
BINARIZATION_TECHNIQUE: str = 'binarizationTechnique'
BULK_DATA = "BulkData_"
COMPLETE_TEXT = "completeText"
COMPLETE_TEXT_EXTENSION = "_completeText.txt"
TEXT_AND_COORDINATES = "textAndCoordinates"
TEXT_AND_COORDINATES_EXTENSION = "_textAndCoordinates.txt"
FILE_TYPE = "fileType"
FILE_PATH = "filePath"
INPUT_PARAMS = "inputParameters"
RESULT_IMAGE = 'result_image'
OPERATION_TYPE = "operationType"
UNIQUE_STRING = "uniqueString"

INPUT_FILE_PATH = "input_file_path"
INSERT = "Insert"
HITS = "hits"
SOURCE = "_source"
PIPELINE_RUNNING_STATUS = "pipelineRunningStatus"
LEFT = "left"
TOP = "top"
WIDTH = "width"
HEIGHT = "height"
X1 = "x1"
Y1 = "y1"
X2 = "x2"
Y2 = "y2"
WORD = "word"
WORDS = "words"
OCR = "OCR"
QUALITY_METRICS = "qualityMetrics"
TEXT = "text"
BLOCKS = "blocks"
STATISTICAL_OPERATION = "statisticalOperation"
NON_STATISTICAL_OPERATION = "nonStatisticalOperation"
DROP_BASED_ON_FORMATS = "dropBasedOnFormats"
DROP_BASED_ON_CONSISTENCY_CHECK = "dropBasedOnConsistencyCheck"
DROP_BASED_ON_PEAKEDNESS = "dropBasedOnPeakedness"
DROP_BASED_ON_SYMMETRY = "dropBasedOnSymmetry"
CHANGE_IMAGE_FORMAT = "changeImageFormat"
CHANGE_INTO_BLACK_AND_WHITE = "changeIntoBlackAndWhite"
DEBLUR = "deBlur"
ADJUST_BRIGHTNESS = "adjustBrightness"
ADJUST_CONTRAST = "adjustContrast"
SET_DPI = "setDPI"
SUPER_RESOLUTION = "superResolution"
ALGO_NAME_FOR_DEBLUR = "algoNameForDeBlur"
MIN_BRIGHTNESS_T = "minBrightnessT"
BRIGHTNESS_VALUE = "brightnessValue"
MIN_CONTRAST_T = "minContrastT"
THRESHOLD_DPI_T = "thresholdDPIT"
DPI_VALUE = "dpiValue"
ALGO_NAME_FOR_SUPER_RESOLUTION = "algoNameForSuperResolution"
RESOLUTION_SCALE = "resolutionScale"
ALGO_TYPE = "algoType"
RESOLUTION_THRESHOLD_MIN_XT = "resolutionThresholdMinXT"
RESOLUTION_THRESHOLD_MIN_YT = "resolutionThresholdMinYT"
THRESHOLD_BLUR_T = "thresholdBlurT"
CONTRAST_VALUE = "contrastValue"
CONTRAST = "contrast"
BRIGHTNESS = "brightness"
IMAGE_ENTROPY = "imageEntropy"
BRISQUARE_SCORE = "brisquareScore"
BLUR_SCORE = "blurScore"
LANGUAGE = "language"
CONTENT_AREA_COVERAGE = "contentAreaCoverage"
VOCAB_COUNT = "vocabCount"
FONT_HEIGHT = "fontHeight"
CONTENT_COUNT = "contentCount"
DETECTING_BLANK_PAGES = "detectingBlankPages"
DETECT_TOO_DARK_IMAGES = "detectTooDarkImages"
DF_CREATION_FLAG = "df_creation_flag"
SUCCESS = "SUCCESS"

SOURCE_OBJ = "source_obj"
INPUT_DICT = "input_dict"
COMMON_DICT = "common_dict"
SINK_DICT = "sink_dict"
LOGS = "logs"
ASSET_DICT = "asset_dict"
INPUT_DATA = "input_data"
RESULT = "result"
DF = "df"
DEV_EXEC_ORDERS = "dev_exec_orders"
SOURCE_DICT = "source_dict"
INTERMEDIATE_SINK_DICT = "intermediate_sink_dict"
PIPELINE_TO_SETTING = "pipeline_to_setting"
IMAGE = "image"
PIPELINE_TO_ID_DETAIL = "pipeline_to_id_detail"
VALUE = "value"
STORAGE_ACCOUNT_VALUE = "ntngidpstorage"
CONTAINER_NAME_VALUE = "idpcontainer"
SAS_TOKEN_VALUE = "?sv=2022-11-02&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2024-03-31T23:59:05Z&st=2024-01-01T15:59:05Z&spr=https,http&sig=tQoGoG7hWGc84nKGXgM%2BU5qA2Ge3jQ5EX0rvJCkHYm8%3D"

# Services
DATA_CLEAN = "Data_Clean"
DATA_EXPLORATION = "Data_Exploration"

# self created JSON tags
SOURCE_INFO = "sourceInfo"
SINK_INFO = "sinkInfo"
ALGO_INFO = "algoInfo"
COMMON_INFO = "commonInfo"
SERVING_INFO = 'servingInfo'

# List of Settings present in the JSON obtained from the UI
SOURCE_SETTINGS: str = 'sourceSettings'
SINK_SETTINGS: str = 'sinkSettings'
DATA_CLEAN_SETTINGS: str = 'dataCleanSettings'
DATA_EXPLORATION_SETTINGS: str = 'dataExplorationSettings'
DEX_ACTION_SETTINGS: str = 'dexActionSettings'
DATA_TRANSFORMATION_SETTINGS: str = 'dataTransformationSettings'
DATA_AUGMENTATION_SETTINGS: str = 'dataAugmentationSettings'
LINKED_NODE_OF_IDP_COMPONENTS: str = 'linkedNodeOfIDPComponents'
INTERMEDIATE_SINK: str = 'intermediateSink'
OCR_SETTINGS = 'ocrSettings'
ZONE_WISE_OCR = "zoneWiseOcr"
MODEL_ALGO_SETTINGS = "modelAlgoSettings"
EVALUATOR_SETTINGS = "evaluatorSettings"
TRAIN_TEST_SPIT_SETTINGS = "trainTestSplitSettings"

MIGRATION_CHECK_SETTINGS = [SOURCE_SETTINGS, SINK_SETTINGS]
DEV_PIPELINE_INFO: str = 'devPipelineInfo'
UPDATED_SINK_SETTING: str = 'updatedsinksetting'
PIPELINE_ORDER: str = 'pipelineOrder'
INFER_PIPELINE_INFO: str = 'inferPipelineInfo'

# keys in linkedNodeOfIDPComponents are -->
PREV_NODE_ID = 'prevNodeId'
NEXT_NODE_ID = 'nextNodeId'
CURRENT_NODE_ID = 'currentNodeId'
SETTINGS = 'settings'

# Common Tags throughout the pipeline
NODE_ID = 'nodeId'
BRANCH_ID: str = 'branchId'

# list of sources and sinks
SOURCE_TYPE = 'sourceType'
WASB: str = 'wasb'
S3: str = 'aws'
G_DRIVE: str = 'googleDrive'
DROP_BOX = 'dropBox'
ONEDRIVE: str = 'oneDrive'

SINK_TYPE = 'sinkType'
WASB_SINK: str = 'wasbSink'

# The List of Constant comes under the common JSON category

USER_ID: str = 'userId'
PIPELINE_ID: str = 'pipelineId'
TIMESTAMP: str = 'timestamp'
VERSION: str = 'version'
LOG_FILE_NAME: str = 'logging_file'
ES_TYPE: str = 'esType'
DEVICE: str = "device"

# common tags for data clean settings
OPERATIONS = 'operations'
INTERMEDIATE_PATH = "intermediatePath"
FILE_COUNT = 'fileCount'
SAMPLE_RESULT_PATH = 'sampleResultPath'
OCR_DATA_INTERMEDIATE_PATH_SUFFIX = 'OCRDataIntermediatePathSuffix'

# The list of DEX tags
STATISTICAL_EXPLORATION = 'statisticalExploration'
NON_STATISTICAL_EXPLORATION = 'nonStatisticalExploration'
DEX_INFO = "dexInfo"
DEX_IN_ACTION = "dexInAction"

# Elastic search tags
ES_HOST = "elasticAddress"
ES_INDEX = "elasticIndex"

# Intermediate Sink Keys
STORAGE_ACCOUNT: str = 'storageAccount'
CONTAINER_NAME: str = 'containerName'
SAS_TOKEN: str = 'sasToken'
OUTPUT_DIR_PATH = 'outputDirectoryPath'
MODEL_OUTPUT_DIR_PATH = 'modelOutputDirectoryPath'

# FTP Source Sink Keys
FTP_HOST: str = 'ftpHost'
FTP_PORT: str = 'ftpPort'
FTP_USER_NAME: str = 'ftpUname'
FTP_PASSWORD: str = 'ftpPass'

IMAGE_DATA: str = 'image_data'
OCR_DATA: str = 'ocr_data'
LABEL_DATA: str = 'labels'
CLASSES_DATA: str = 'classes'
TTS_TYPE: str = 'TTS_Type'

# Data Transformation Keys

MAPPING_OPERATION: str = 'mappingOperation'

""" ###### Serving Constants ###### """
MODEL_INFERENCE: str = 'modelInference'
LINKED_DEV_PIPELINE_DETAIL: str = 'linkedDevPipelineDetail'
PIPELINE_NAME: str = 'pipelineName'
DATA_FRAME = 'df'
IS_INITIAL_PIPELINE = 'initial_pipeline'
MAPPING = 'mapping'
POST_PROCESSING = 'postProcessing'
SECTION = 'section'
POST_PROCESSING_NODES = 'nodes'
COMMON_JSON = 'common_json'
PREDICTED_CLASS = 'predicted_class'
CONFIDENCE = 'confidence'
CONFIDENCE_CONDITION = 'confidenceCondition'
TAIL_PIPELINE = 'tailPipeline'
IMAGE_NAME = 'image_name'
RESULT = 'result'
SERVING_METADATA = 'servingMetadata'
INFERENCE_RESULT = 'inferenceResult'

# Pipeline node status
FINAL_STATUS = "finalStatus"
NODE_STATUS = "nodeStatus"
STATUS_RUNNING = "running"
STATUS_BREAK = "break"
STATUS_COMPLETED = "completed"
STATUS_PENDING = "pending"

IS_SERVING_APPLIED: str = "isServingApplied"

""" Tags related to Image Information """

MIME_TYPE: str = 'mimeType'
ID: str = 'id'
NAME: str = 'name'
SIZE: str = 'size'
PATH: str = 'path'
PNG_EXTN: str = 'png'

""" Source, Sink, Intermediate Sink related constants """

# G Drive tags
FOLDER_ID: str = 'folderId'
PARENT_FOLDER_NAME: str = 'parentFolderName'
ACCESS_TOKEN: str = 'accessToken'
META_DATA: str = 'metaData'

# MIME Types
PNG_IMAGE: str = 'image/png'
JPEG_IMAGE: str = 'image/jpeg'
JPG_IMAGE: str = 'image/jpg'
TEXT_PLAIN: str = "text/plain"
FOLDER_TYPE: str = 'application/vnd.google-apps.folder'

# Drop Box Source's tags
FILE_TAG: str = 'content_hash'
FILE_TAG_VALUE = None
DBX_NAME: str = 'name'

# S3 Source's tags
AWS_ACCESS_KEY_ID: str = 'awsKeyId'
AWS_SECRET_ACCESS_KEY: str = 'awsKeySecret'
AWS_BUCKET_NAME: str = 'awsBucketName'
AWS_CLIENT: str = 's3'
AWS_ITEM_NAME: str = 'name'
AWS_FILE_TYPE: str = 'file'
AWS_FOLDER_TYPE: str = 'folder'
AWS_TYPE: str = 'type'
AWS_OBJ_NAME: str = 'Key'
AWS_CONTENTS: str = 'Contents'
AWS_IMG_RESPONSE_TAG: str = 'Body'

# Jupyter Notebook protocol related tags
END_POINT: str = "endPoint"
PIPELINE_RESULT: str = 'pipeline_result'
OTHER_PARAMS: str = 'other_params'
BASE_64_ENCODED_IMAGE: str = "b64_encoded_image"

# Inference Related Tags
DEV_EXECUTION_ORDER: str = "devExecOrder"
INFER_ORDER: str = "infer_order"
UNCOMPRESSED_IMAGES: str = 'uncompressed_images'

# Production Related Constants
SCHEDULING_FLAG: str = 'scheduling'
PRODUCTION_ES_TYPE: str = 'productionDetails'
MODEL_PATH: str = 'modelPath'
PRODUCTION_BEST_MODEL_PATH: str = 'allBestModelsInfo'

IS_LOAD_MODEL: str = 'isLoadModel'
HTTP_REQUEST: str = "httpRequest"

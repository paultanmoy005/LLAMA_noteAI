#This query returns distinct event id and updated date time pair for notes with sexual orientation related kewords
WITH A AS
(
    SELECT DISTINCT BLOB.EVENT_ID, BLOB.VALID_UNTIL_DT_TM, CE.PARENT_EVENT_ID, CE.UPDT_DT_TM,
    ROW_NUMBER() OVER (PARTITION BY CE.PARENT_EVENT_ID ORDER BY CE.UPDT_DT_TM DESC) AS RN
    FROM MILLENNIUM.V500.CE_BLOB_DECOMPRESSED_VALID AS BLOB
    JOIN MILLENNIUM.V500.CLINICAL_EVENT AS CE
    ON CE.EVENT_ID = BLOB.EVENT_ID
    WHERE LOWER(DECOMPRESSED_NOTE) LIKE '%lesbian%'
    OR LOWER(DECOMPRESSED_NOTE) LIKE '%homosexual%'
    OR LOWER(DECOMPRESSED_NOTE) LIKE '%heterosexual%'
    OR LOWER(DECOMPRESSED_NOTE) LIKE '%bisexual%'
)
SELECT DISTINCT EVENT_ID, UPDT_DT_TM
FROM A WHERE RN=1;

-- ============================================
-- CHECK SPECTROGRAM URL STORAGE IN SNOWFLAKE
-- Run these queries to verify spectrogram URLs are being stored
-- ============================================

-- STEP 1: Check your current context
SELECT CURRENT_DATABASE(), CURRENT_SCHEMA();

-- STEP 2: Verify the SPECTROGRAM_URL column exists
-- This will show all columns in the table
DESCRIBE TABLE MONITORING_SESSIONS;

-- STEP 3: Check if SPECTROGRAM_URL column exists (look for it in the output above)
-- Or use this query to check specifically:
SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'MONITORING_SESSIONS'
  AND COLUMN_NAME = 'SPECTROGRAM_URL';

-- STEP 4: View recent records with spectrogram URLs
-- Shows the last 10 records with their spectrogram URLs
SELECT 
    SESSION_ID,
    TIMESTAMP,
    FATIGUE_STATUS,
    FILENAME as AUDIO_URL,
    SPECTROGRAM_URL,
    CASE 
        WHEN SPECTROGRAM_URL IS NOT NULL THEN 'YES'
        ELSE 'NO'
    END as HAS_SPECTROGRAM
FROM MONITORING_SESSIONS
ORDER BY TIMESTAMP DESC
LIMIT 10;

-- STEP 5: Count how many records have spectrogram URLs
SELECT 
    COUNT(*) as TOTAL_RECORDS,
    COUNT(SPECTROGRAM_URL) as RECORDS_WITH_SPECTROGRAM,
    COUNT(*) - COUNT(SPECTROGRAM_URL) as RECORDS_WITHOUT_SPECTROGRAM,
    ROUND(COUNT(SPECTROGRAM_URL) * 100.0 / COUNT(*), 2) as PERCENTAGE_WITH_SPECTROGRAM
FROM MONITORING_SESSIONS;

-- STEP 6: Show sample spectrogram URLs (if any exist)
SELECT 
    SESSION_ID,
    TIMESTAMP,
    SPECTROGRAM_URL
FROM MONITORING_SESSIONS
WHERE SPECTROGRAM_URL IS NOT NULL
ORDER BY TIMESTAMP DESC
LIMIT 5;

-- STEP 7: If the column doesn't exist, add it:
-- ALTER TABLE MONITORING_SESSIONS ADD COLUMN IF NOT EXISTS SPECTROGRAM_URL VARCHAR(500);


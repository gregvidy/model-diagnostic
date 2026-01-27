WITH cte_base AS (
    SELECT 
        Transaction_Serial_No,
        COALESCE(PANNumber, "-999") AS PANNumber,
        COALESCE(CurrencyCode, "-999") AS CurrencyCode,
        COALESCE(MCC, "-999") AS MCC,
        Transaction_Datetime
    FROM C03_Channel

    UNION ALL

    SELECT 
        Transaction_Serial_No,
        COALESCE(PANNumber, "-999") AS PANNumber,
        COALESCE(CurrencyCode, "-999") AS CurrencyCode,
        COALESCE(MCC, "-999") AS MCC,
        Transaction_Datetime
    FROM C09_Channel
),

cte_final AS (
    SELECT 
        A.Transaction_Serial_No,
        A.PANNumber,
        A.CurrencyCode,
        A.MCC,
        A.Transaction_Datetime,

        -- Rolling unique MCCs per PANNumber
        (
            SELECT COUNT(DISTINCT B.MCC)
            FROM cte_base AS B
            WHERE 
                B.PANNumber = A.PANNumber AND
                B.Transaction_Datetime < A.Transaction_Datetime AND
                B.Transaction_Datetime > DATEADD(MINUTE, -15, A.Transaction_Datetime)
        ) AS CntUniqueMCCByCardNo_L15min,

        (
            SELECT COUNT(DISTINCT B.MCC)
            FROM cte_base AS B
            WHERE 
                B.PANNumber = A.PANNumber AND
                B.Transaction_Datetime < A.Transaction_Datetime AND
                B.Transaction_Datetime > DATEADD(DAY, -30, A.Transaction_Datetime)
        ) AS CntUniqueMCCByCardNo_L30D,

        -- Rolling unique PANNumbers per CurrencyCode
        (
            SELECT COUNT(DISTINCT B.PANNumber)
            FROM cte_base AS B
            WHERE 
                B.CurrencyCode = A.CurrencyCode AND
                B.Transaction_Datetime < A.Transaction_Datetime AND
                B.Transaction_Datetime > DATEADD(MINUTE, -15, A.Transaction_Datetime)
        ) AS CntUniqueCardNoByCurrencyCode_L15min,

        (
            SELECT COUNT(DISTINCT B.PANNumber)
            FROM cte_base AS B
            WHERE 
                B.CurrencyCode = A.CurrencyCode AND
                B.Transaction_Datetime < A.Transaction_Datetime AND
                B.Transaction_Datetime > DATEADD(DAY, -30, A.Transaction_Datetime)
        ) AS CntUniqueCardNoByCurrencyCode_L30D

    FROM cte_base AS A
)

SELECT
    Transaction_Serial_No,
    PANNumber,
    CntUniqueCardNoByCurrencyCode_L30D,
    -- Safe division: if denominator is 0 or NULL, impute ratio as -999
    ISNULL(
        CntUniqueCardNoByCurrencyCode_L30D / 
        NULLIF(CntUniqueCardNoByCurrencyCode_L15min, 0),
        -999        
    ) AS RatioCntUniqueCardNoByCurrencyCode_L30DL15min,

    CntUniqueMCCByCardNo_L15min,
    -- Safe division: if denominator is 0 or NULL, impute ratio as -999
    ISNULL(
        CntUniqueMCCByCardNo_L30D / 
        NULLIF(CntUniqueMCCByCardNo_L15min, 0),
        -999        
    ) AS RatioCntUniqueMCCByCardNo_L30DL15min,
FROM cte_final
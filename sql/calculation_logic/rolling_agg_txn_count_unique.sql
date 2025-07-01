WITH cte_base AS (
    SELECT 
        Transaction_Serial_No,
        PANNumber,
        CurrencyCode,
        MCC,
        Transaction_Datetime
    FROM C03_Channel

    UNION ALL

    SELECT 
        Transaction_Serial_No,
        PANNumber,
        CurrencyCode,
        MCC,
        Transaction_Datetime
    FROM C09_Channel
)

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
            B.Transaction_Datetime >= DATEADD(MINUTE, -15, A.Transaction_Datetime)
    ) AS CntUnique_MCC_by_CardNo_L15M,

    (
        SELECT COUNT(DISTINCT B.MCC)
        FROM cte_base AS B
        WHERE 
            B.PANNumber = A.PANNumber AND
            B.Transaction_Datetime < A.Transaction_Datetime AND
            B.Transaction_Datetime >= DATEADD(DAY, -30, A.Transaction_Datetime)
    ) AS CntUnique_MCC_by_CardNo_L30D,

    -- Rolling unique PANNumbers per CurrencyCode
    (
        SELECT COUNT(DISTINCT B.PANNumber)
        FROM cte_base AS B
        WHERE 
            B.CurrencyCode = A.CurrencyCode AND
            B.Transaction_Datetime < A.Transaction_Datetime AND
            B.Transaction_Datetime >= DATEADD(MINUTE, -15, A.Transaction_Datetime)
    ) AS CntUnique_CardNo_by_Currency_L15M,

    (
        SELECT COUNT(DISTINCT B.PANNumber)
        FROM cte_base AS B
        WHERE 
            B.CurrencyCode = A.CurrencyCode AND
            B.Transaction_Datetime < A.Transaction_Datetime AND
            B.Transaction_Datetime >= DATEADD(DAY, -30, A.Transaction_Datetime)
    ) AS CntUnique_CardNo_by_Currency_L30D

FROM cte_base AS A;

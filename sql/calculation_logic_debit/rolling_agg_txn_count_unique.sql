WITH cte_base AS (
    SELECT
        Transaction_Serial_No,
        COALESCE(Debit_No, '-999') AS Debit_No,
        COALESCE(MCC, '-999') AS MCC,
        Transaction_Datetime,
    FROM C06_Channel
    UNION ALL
    SELECT
        Transaction_Serial_No,
        COALESCE(Debit_No, '-999') AS Debit_No,
        COALESCE(MCC, '-999') AS MCC,
        Transaction_Datetime,
    FROM C10_Channel
),

cte_final AS (
    SELECT 
        A.Transaction_Serial_No,
        A.Debit_No,
        A.MCC,
        A.Transaction_Datetime,

        -- Rolling unique Debit_Nos per MCC
        (
            SELECT COUNT(DISTINCT B.Debit_No)
            FROM cte_base AS B
            WHERE 
                B.MCC = A.MCC AND
                B.Transaction_Datetime < A.Transaction_Datetime AND
                B.Transaction_Datetime >= DATEADD(MINUTE, -15, A.Transaction_Datetime)
        ) AS CntUniqueCardNoByMCC_L15min,

        (
            SELECT COUNT(DISTINCT B.MCC)
            FROM cte_base AS B
            WHERE 
                B.MCC = A.MCC AND
                B.Transaction_Datetime < A.Transaction_Datetime AND
                B.Transaction_Datetime >= DATEADD(DAY, -30, A.Transaction_Datetime)
        ) AS CntUniqueCardNoByMCC_L30D,

    FROM cte_base AS A
)

SELECT
    Transaction_Serial_No,
    Debit_No,
    CntUniqueCardNoByMCC_L15min,
    CntUniqueCardNoByMCC_L30D,

    -- Safe division: if denominator is 0 or NULL, impute ratio as 0
    ISNULL(
        CntUniqueCardNoByMCC_L30D / 
        NULLIF(CntUniqueCardNoByMCC_L15min, 0),
        0        
    ) AS RatioCntUniqueCardNoByMCC_L30DL15min,

FROM cte_final
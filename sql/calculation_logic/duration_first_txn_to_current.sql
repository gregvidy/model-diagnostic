WITH cte_base AS (
    SELECT
        Transaction_Serial_No,
        PANNumber,
        MCC,
        Transaction_Datetime
    FROM C03_Channel

    UNION ALL

    SELECT
        Transaction_Serial_No,
        PANNumber,
        MCC,
        Transaction_Datetime
    FROM C09_Channel
),

t_min_date AS (
    SELECT
        PANNumber,
        MCC,
        MIN(Transaction_Datetime) AS first_txn_date
    FROM cte_base
    GROUP BY PANNumber, MCC
),

cte_joined AS (
    SELECT
        A.Transaction_Serial_No,
        A.PANNumber,
        A.MCC,
        A.Transaction_Datetime,
        DATEDIFF(DAY, B.first_txn_date, A.Transaction_Datetime) AS TimeFirstTxnToCurrentMCC
    FROM cte_base AS A
    LEFT JOIN t_min_date AS B
        ON A.PANNumber = B.PANNumber
        AND A.MCC = B.MCC
)

SELECT 
    A.Transaction_Serial_No,
    A.PANNumber,
    A.MCC,
    A.Transaction_Datetime,
    A.TimeFirstTxnToCurrentMCC,

    -- Rolling avg over last 15 mins, excluding current row
    (
        SELECT AVG(B.TimeFirstTxnToCurrentMCC * 1.0)
        FROM cte_joined AS B
        WHERE 
            B.PANNumber = A.PANNumber AND
            B.MCC = A.MCC AND
            B.Transaction_Datetime < A.Transaction_Datetime AND
            B.Transaction_Datetime >= DATEADD(MINUTE, -15, A.Transaction_Datetime)
    ) AS AvgTimeFirstTxnToCurrentMCC_Last15Mins,

    -- Rolling avg over last 30 days, excluding current row
    (
        SELECT AVG(B.TimeFirstTxnToCurrentMCC * 1.0)
        FROM cte_joined AS B
        WHERE 
            B.PANNumber = A.PANNumber AND
            B.MCC = A.MCC AND
            B.Transaction_Datetime < A.Transaction_Datetime AND
            B.Transaction_Datetime >= DATEADD(DAY, -30, A.Transaction_Datetime)
    ) AS AvgTimeFirstTxnToCurrentMCC_Last30D

FROM cte_joined AS A;

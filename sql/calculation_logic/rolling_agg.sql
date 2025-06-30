WITH cte_base AS (
    SELECT 
        Transaction_id,
        PANNumber,
        MCC,
        Transaction_datetime,
        Transaction_amount
    FROM C03_channel
)

SELECT 
    A.Transaction_id,
    A.PANNumber,
    A.MCC,
    A.Transaction_datetime,
    A.Transaction_amount,

    -- 15-minute window
    (
        SELECT COUNT(*) 
        FROM cte_base AS B
        WHERE 
            B.MCC = A.MCC AND
            B.PANNumber = A.PANNumber AND
            B.Transaction_datetime < A.Transaction_datetime AND
            B.Transaction_datetime >= DATEADD(MINUTE, -15, A.Transaction_datetime)
    ) AS rolling_count_15m_excl,

    (
        SELECT SUM(B.Transaction_amount)
        FROM cte_base AS B
        WHERE 
            B.MCC = A.MCC AND
            B.PANNumber = A.PANNumber AND
            B.Transaction_datetime < A.Transaction_datetime AND
            B.Transaction_datetime >= DATEADD(MINUTE, -15, A.Transaction_datetime)
    ) AS rolling_sum_15m_excl,

    (
        SELECT AVG(B.Transaction_amount)
        FROM cte_base AS B
        WHERE 
            B.MCC = A.MCC AND
            B.PANNumber = A.PANNumber AND
            B.Transaction_datetime < A.Transaction_datetime AND
            B.Transaction_datetime >= DATEADD(MINUTE, -15, A.Transaction_datetime)
    ) AS rolling_avg_15m_excl,

    (
        SELECT MAX(B.Transaction_amount)
        FROM cte_base AS B
        WHERE 
            B.MCC = A.MCC AND
            B.PANNumber = A.PANNumber AND
            B.Transaction_datetime < A.Transaction_datetime AND
            B.Transaction_datetime >= DATEADD(MINUTE, -15, A.Transaction_datetime)
    ) AS rolling_max_15m_excl,

    -- 30-day window
    (
        SELECT COUNT(*) 
        FROM cte_base AS B
        WHERE 
            B.MCC = A.MCC AND
            B.PANNumber = A.PANNumber AND
            B.Transaction_datetime < A.Transaction_datetime AND
            B.Transaction_datetime >= DATEADD(DAY, -30, A.Transaction_datetime)
    ) AS rolling_count_30d_excl,

    (
        SELECT SUM(B.Transaction_amount)
        FROM cte_base AS B
        WHERE 
            B.MCC = A.MCC AND
            B.PANNumber = A.PANNumber AND
            B.Transaction_datetime < A.Transaction_datetime AND
            B.Transaction_datetime >= DATEADD(DAY, -30, A.Transaction_datetime)
    ) AS rolling_sum_30d_excl,

    (
        SELECT AVG(B.Transaction_amount)
        FROM cte_base AS B
        WHERE 
            B.MCC = A.MCC AND
            B.PANNumber = A.PANNumber AND
            B.Transaction_datetime < A.Transaction_datetime AND
            B.Transaction_datetime >= DATEADD(DAY, -30, A.Transaction_datetime)
    ) AS rolling_avg_30d_excl,

    (
        SELECT MAX(B.Transaction_amount)
        FROM cte_base AS B
        WHERE 
            B.MCC = A.MCC AND
            B.PANNumber = A.PANNumber AND
            B.Transaction_datetime < A.Transaction_datetime AND
            B.Transaction_datetime >= DATEADD(DAY, -30, A.Transaction_datetime)
    ) AS rolling_max_30d_excl

FROM cte_base AS A;

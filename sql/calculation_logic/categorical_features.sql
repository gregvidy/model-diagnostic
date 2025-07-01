WITH cte_base AS (
    SELECT
        Transaction_Serial_No,
        COALESCE(PANNumber, '-999') AS PANNumber,
        COALESCE(MCC, '-999') AS MCC,
        Transaction_Datetime,
        Transaction_Amount,
    FROM C03_Channel
    UNION ALL
    SELECT
        Transaction_Serial_No,
        COALESCE(PANNumber, '-999') AS PANNumber,
        COALESCE(MCC, '-999') AS MCC,
        Transaction_Datetime,
        Transaction_Amount,
    FROM C09_Channel
),

SELECT
    cte_base.Transaction_Serial_No,
    cte_base.PANNumber
    CASE
        WHEN tscf.CustomerSex IS NULL THEN "__missing__"
        WHEN tscf.CustomerSex NOT IN (0, 1, 2) THEN "__missing__"
    ELSE tscf.CustomerSex
    END AS CustomerSex,
    COALESCE(cte_base.Transaction_Amount, 0) AS Transaction_Amount,
    COALESCE(tscf.TotalTrxAmount10Mi, 0) AS TotalTrxAmount10Mi,
    mrt.MCC_Category
FROM cte_base
LEFT JOIN Transaction_Summary_Calculation_Fraud tscf
    ON tscf.Transaction_Serial_No = cte_base.Transaction_Serial_No
LEFT JOIN MCC_Reference_Table AS mrt
    ON mrt.MCC = cte_base.MCC
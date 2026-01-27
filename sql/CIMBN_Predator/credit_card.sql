/*
This logic only applies from July 2025 - Now
1. CTE Channel -> full outer join between C01 and C05
2. CTE TSCF -> filter which calc. variables that used in modelling
3. CTE TSF -> take only label and key ID (Transaction_Serial_No)
4. CTE Down-Sampling
5. Final query -> list of raw features
*/

/*
before July 2025
*/

-- GOAL: from analysis you get the final feature set for new model
-- -> evaluate: existing EM features, current calculation variables in TSCF
-- -> create: new strong features for ML model -> incorporate MerchantID, MCC
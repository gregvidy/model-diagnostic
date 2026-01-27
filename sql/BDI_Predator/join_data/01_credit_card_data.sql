-- Variable settings:
-- In this query, we sampling down the Clean transactions and deciding the start and end date
DECLARE @sampling_percentage FLOAT = {sampling_pct};
DECLARE @sampling_start_date DATE = '{start_date}';
DECLARE @sampling_end_date DATE = '{end_date}';

-- 3 tables that used in the query:
-- Transaction_Summary_Fraud_Hashed
-- C03_Details (channel)
-- C09_Details (channel)
-- Transaction_Summary_Calculations_Fraud_Hashed
WITH tsf_credit AS (
    SELECT
        Transaction_Serial_No
        , Confirmed
    FROM Transaction_Summary_Fraud_Hashed
    WHERE Channel = 'C03'
    OR Channel = 'C09'
)

-- Calling all the channel columns that will be used for Feature Engineering later
, t_union_base AS (
    SELECT
		c03.PANNumber
        , c03.[Transaction Serial No]
		, c03.[Transaction Datetime]
		, c03.[Product Indicator]
		, c03.[Transaction Amount]
		, c03.MCC
		, c03.[Country Code]
		, c03.[Card Acceptor TerminalID] AS 'Card Acceptor Terminal ID'
		, c03.[Card Acceptor ID]
		, c03.[Terminal Owner] AS 'Card Acceptor Name'
		, c03.[Terminal CIty] AS 'Card Acceptor City'
		, c03.[Terminal State] AS 'Card Acceptor Region Code'
		, c03.[Terminal Country] AS 'Card Acceptor Country Code'
		, c03.[Currency Code]
        , tsf_credit.Confirmed
    FROM C03_Details_Hashed AS c03
    LEFT JOIN tsf_credit
        ON c03.[Transaction Serial No] = tsf_credit.[Transaction_Serial_No]

    UNION ALL

    SELECT
		c09.[Card No] AS PANNumber
        , c09.Transaction_Serial_No AS 'Transaction Serial No'
		, c09.[Transaction Datetime]
		, NULL AS 'Product Indicator'
		, c09.[Transaction Amount IDR] AS 'Transaction Amount'
		, c09.MCC
		, c09.[Country Code]
		, c09.[Card Acceptor Terminal ID]
		, c09.[Card Acceptor Id Code] AS 'Card Acceptor ID'
		, c09.[Card Acceptor Name]
		, c09.[Card Acceptor City]
		, c09.[Card Acceptor Region Code]
		, c09.[Card Acceptor Country Code]
		, c09.[Transaction Currency Code] AS 'Currency Code'
        , tsf_credit.Confirmed
    FROM C09_Details_Hashed AS c09
    LEFT JOIN tsf_credit
        ON c09.[Transaction_Serial_No] = tsf_credit.[Transaction_Serial_No]
)

-- Sampling down the clean transactions
, t_base AS (
    SELECT *
    FROM t_union_base
    WHERE [Transaction Datetime] >= CAST(@sampling_start_date AS DATE)
    AND [Transaction Datetime] <= CAST(@sampling_end_date AS DATE)
)
    
, t_list_card_ever_alert AS (
    SELECT DISTINCT PANNumber
    FROM t_base
    WHERE Confirmed = 1 OR Confirmed = 0
)

, t_list_card_never_alert AS (
    SELECT DISTINCT t_base.PANNumber
    FROM t_base
    LEFT JOIN t_list_card_ever_alert
        ON t_base.PANNumber = t_list_card_ever_alert.PANNumber
    WHERE t_list_card_ever_alert.PANNumber IS NULL
)

, t_sample AS (
    SELECT PANNumber
    FROM (
        SELECT
            PANNumber
            , ROW_NUMBER() OVER (ORDER BY NEWID()) AS rn
            , COUNT(*) OVER () AS total
        FROM t_list_card_never_alert
    ) AS sub
    WHERE rn <= total * @sampling_percentage
)

, t_population_ever_alert AS (
	SELECT
		PANNumber
        , [Transaction Serial No]
		, [Transaction Datetime]
		, [Product Indicator]
		, [Transaction Amount]
		, MCC
		, [Country Code]
		, [Card Acceptor Terminal ID]
		, [Card Acceptor ID]
        , [Card Acceptor Name]
        , [Card Acceptor City]
        , [Card Acceptor Region Code]
        , [Card Acceptor Country Code]
		, [Currency Code]
		, Confirmed
	FROM t_base
	WHERE Confirmed = 1 OR Confirmed = 0
    UNION ALL
    SELECT
		t_base.PANNumber
        , [Transaction Serial No]
		, [Transaction Datetime]
		, [Product Indicator]
		, [Transaction Amount]
		, MCC
		, [Country Code]
		, [Card Acceptor Terminal ID]
		, [Card Acceptor ID]
        , [Card Acceptor Name]
        , [Card Acceptor City]
        , [Card Acceptor Region Code]
        , [Card Acceptor Country Code]
		, [Currency Code]
		, Confirmed
    FROM t_base
    INNER JOIN t_list_card_ever_alert
        ON t_base.PANNumber = t_list_card_ever_alert.PANNumber
    WHERE Confirmed IS NULL
)

, t_population_all_clean AS (
    SELECT
		t_base.PANNumber
        , [Transaction Serial No]
		, [Transaction Datetime]
		, [Product Indicator]
		, [Transaction Amount]
		, MCC
		, [Country Code]
		, [Card Acceptor Terminal ID]
		, [Card Acceptor ID]
        , [Card Acceptor Name]
        , [Card Acceptor City]
        , [Card Acceptor Region Code]
        , [Card Acceptor Country Code]
		, [Currency Code]
		, Confirmed
    FROM t_base
    INNER JOIN t_sample
        ON t_base.PANNumber = t_sample.PANNumber
)

, t_final AS (
	SELECT * FROM t_population_ever_alert
	UNION ALL
	SELECT * FROM t_population_all_clean
)

SELECT
    t_final.PANNumber
	, t_final.[Transaction Serial No]
    , t_final.[Transaction Datetime]
    , t_final.[Product Indicator]
    , t_final.[Transaction Amount]
    , t_final.MCC
    , t_final.[Country Code]
    , t_final.[Card Acceptor Terminal ID]
    , t_final.[Card Acceptor ID]
    , t_final.[Card Acceptor Name]
    , t_final.[Card Acceptor City]
    , t_final.[Card Acceptor Region Code]
    , t_final.[Card Acceptor Country Code]
    , t_final.[Currency Code]
	, tscf.CustomerAge
	, tscf.AccountStatus
	, tscf.CustomerAvgIncome
	, tscf.CardStatus
	, tscf.Balance
	, tscf.AgeOfOpenAcctTxn
	, tscf.AgeOfActiveCardTxn
	, tscf.AgeOfOpenAcctActiveCard
	, tscf.TrfToBDIStaff
	, tscf.BDIStaff
	, tscf.AgeOfRegDateTxn
	, tscf.IsSDBPastDue
	, tscf.FlagOutBranch
	, tscf.isTDHoldAmount
	, tscf.isProgramHoldAmount
	, tscf.IsOfficeHour
	, tscf.VALAS
	, tscf.ChannelTrx
	, tscf.LastTrxAmount
	, tscf.DrLastTrxAmount
	, tscf.CrLastTrxAmount
	, tscf.TotalTrxAmountL1D
	, tscf.DrTotalTrxAmountL1D
	, tscf.CrTotalTrxAmountL1D
	, tscf.TotalTrxAmountL7D
	, tscf.DrTotalTrxAmountL7D
	, tscf.CrTotalTrxAmountL7D
	, tscf.TotalTrxAmountL1M
	, tscf.DrTotalTrxAmountL1M
	, tscf.CrTotalTrxAmountL1M
	, tscf.AvgTrxAmountL1D
	, tscf.AvgTrxAmountL7D
	, tscf.AvgTrxAmountL1M
	, tscf.TotalTrxAmount15Mi
	, tscf.HIghRiskCustomer
	, tscf.ProductName
	, tscf.PrevAccStatus
	, tscf.HoldAmountTB
	, tscf.HoldAmountDP
	, tscf.SDBBalance
	, tscf.CCBrnAccount
	, tscf.CCBrnReactivedAccDormant
	, tscf.BrnNameReactivedAccDormant
	, tscf.TrxBranchName
	, tscf.IsWhiteList
	, tscf.IsBlacklistAccount
	, tscf.IsBlacklistCust
	, tscf.ProductCodeTo
	, tscf.AccessCode
	, tscf.SDBBoxNo
	, tscf.SDBStatus
	, tscf.SDBLastPaymentAmount
	, tscf.FlagTxnDesc
	, tscf.ProductCodeDep
	, tscf.IsMerchantBDI
	, tscf.NumOfTrxL1D
	, tscf.NumOfTrxL7D
	, tscf.IsHighRiskCountry
	, tscf.IsBlacklistMerchant
	, tscf.IsWatchListMerchant
	, tscf.IsHighRiskMCC
	, tscf.CreditLimit
	, tscf.CashCreditLimit
	, tscf.CardProduct
	, tscf.CustomerSex
	, tscf.IsBlacklistCardNo
	, tscf.IsValidCard
	, tscf.Currency
	, tscf.POSMode
	, tscf.PINEntryCapability
	, tscf.IsCardExpired
	, tscf.Is3DSecure
	, tscf.IsWhiteListMerchant
	, tscf.IsBDIEmail
	, tscf.IsWhiteListCardNo
	, tscf.IsWhiteListAccountNo
	, tscf.IsWatchListAccountNo
	, tscf.IsAccountBDIEmployee
	, tscf.CodAccess
	, tscf.TransactionType
	, tscf.TotalTrxAmount10Mi
	, tscf.CustomerType
	, tscf.TotalTrxAmountL5min
	, tscf.TotalTrxAmountContactless
	, tscf.IsWhitelistCardNoOpen
	, tscf.DaysOfWhitelistCardNoOpen
	, tscf.IsVAAccount
	, tscf.IsExcludeTeminalId
	, tscf.isBDICard
	, tscf.IsWatchlistCardNo
	, tscf.SourceChannel
	, tscf.IsBlackListTID
	, tscf.IsFirstTrx3DS
	, tscf.VCardDataInputCapability
	, tscf.CustomerNationality
	, tscf.TotalTrxAmountContactlessPerDay
	, tscf.IsTopUp
	, tscf.IsBillPayment
	, tscf.IsPaymentToVA
	, tscf.TotalTrxAmountTopUp1
	, tscf.TotalTrxAmountTopUp2
	, tscf.TotalTrxAmountVA1
	, tscf.TotalTrxAmountVA2
	, tscf.CustomerEwaletNumber
	, tscf.IsHighRiskCountryThresholdAmt
	, tscf.IsHighRiskCurrency
	, tscf.IsHighRiskCurrencyThresholdAmt
	, tscf.IsQRtrx
	, tscf.IsDCashTrx
	, tscf.IsFTTrx
	, tscf.TotalTrxAmountQRIS
	, tscf.TotalTrxAmountDcash
	, tscf.TotalTrxAmountTrf
	, tscf.CountTrxQRIS
	, tscf.CountTrxTrf
	, tscf.TotalTrxAmountTrf2
	, tscf.CountTrxQRIS2
	, tscf.CountTrxDcash
	, tscf.CountTrxTrf2
	, tscf.NumOfTrxL15Min
	, tscf.IsWhitelistCardNoPrevilage
	, tscf.IsWhiteListTID
	, tscf.IsWatchListTID
	, tscf.TimeDiffCurrPrev
	, tscf.TotalTrxAmount30Mi
	, tscf.IsBucket1WatchListCardNo
	, tscf.IsBucket1BlacklistCardNo
	, tscf.IsBucket1WhiteListCardNo
	, tscf.IsBucket2WhiteListMerchant
	, tscf.IsBucket2WatchListMerchant
	, tscf.IsBucket2BlacklistMerchant
	, tscf.CountTrxEOD
	, t_final.Confirmed
FROM t_final
LEFT JOIN Transaction_Summary_Calculations_Fraud_Hashed tscf
	ON t_final.[Transaction Serial No] = tscf.[Transaction_Serial_No];
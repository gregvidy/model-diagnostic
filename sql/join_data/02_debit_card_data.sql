DECLARE @sampling_percentage FLOAT = {sampling_pct};

WITH tsf_credit AS (
    SELECT *
    FROM Transaction_Summary_Fraud
    WHERE Channel = 'C06'
    AND Channel = 'C09'
)

, t_base AS (
    SELECT
		c06.PANNumber
        , c06.[Transaction_Serial_No] AS 'Transaction Serial No'
		, c06.[Transaction Datetime]
		, c06.[Transaction Amount]
		, c06.MCC
		, c06.CardAcceptorCountryCode AS 'Country Code'
		, c06.CardAcceptorTerminalID AS 'Card Acceptor Terminal ID'
		, c06.CardAcceptorCity AS 'Card Acceptor City' 
		, c06.CardAcceptorName AS 'Card Acceptor Name' 
		, c06.CardAcceptorRegionCode AS 'Card Acceptor Region' 
		, c06.CardAcceptorCountryCode AS 'Card Acceptor Country Code' 
		, c06.TransactionCurrencyCode AS 'Currency Code'
        , tsf_credit.Confirmed
    FROM C06_Details AS c06
    LEFT JOIN tsf_credit
        ON c06.[Transaction Serial No] = tsf_credit.[Transaction Serial No]

    UNION ALL

    SELECT
		c10.PANNumber
        , c10.[Transaction_Serial_No] AS 'Transaction Serial No'
		, c10.[Transaction Datetime]
		, c10.[Transaction Amount]
		, c10.MCC
		, c10.CardAcceptorCountryCode AS 'Country Code'
		, c10.CardAcceptorTerminalID AS 'Card Acceptor Terminal ID'
		, c10.CardAcceptorCity AS 'Card Acceptor City' 
		, c10.CardAcceptorName AS 'Card Acceptor Name' 
		, c10.CardAcceptorRegionCode AS 'Card Acceptor Region' 
		, c10.CardAcceptorCountryCode AS 'Card Acceptor Country Code' 
		, '360' AS 'Currency Code'
        , tsf_credit.Confirmed
    FROM C10_Details AS c10
    LEFT JOIN tsf_credit
        ON c10.[Transaction_Serial_No] = tsf_credit.[Transaction Serial No]
)

, t_clean AS (
	SELECT
		PANNumber
        , [Transaction Serial No]
		, [Transaction Datetime]
		, [Transaction Amount]
		, MCC
		, [Country Code]
		, [Card Acceptor Terminal ID]
        , [Card Acceptor Name]
        , [Card Acceptor City]
        , [Card Acceptor Region]
        , [Card Acceptor Country Code]
		, [Currency Code]
		, Confirmed
	FROM (
		SELECT *,
			ROW_NUMBER() OVER(
				PARTITION BY PANNumber
				ORDER BY CHECKSUM([Transaction Serial No])
			) AS rn,
			COUNT(*) OVER(
				PARTITION BY PANNumber
			) AS total
		FROM t_base
		WHERE Confirmed = 0 OR Confirmed IS NULL
	) AS sub
	WHERE rn <= total * (@sampling_percentage)
)

, t_fraud AS (
	SELECT
		PANNumber
        , [Transaction Serial No]
		, [Transaction Datetime]
		, [Transaction Amount]
		, MCC
		, [Country Code]
		, [Card Acceptor Terminal ID]
        , [Card Acceptor Name]
        , [Card Acceptor City]
        , [Card Acceptor Region]
        , [Card Acceptor Country Code]
		, [Currency Code]
		, Confirmed
	FROM t_base
	WHERE Confirmed = 1
)

, t_final AS (
	SELECT * FROM t_clean
	UNION ALL
	SELECT * FROM t_fraud
)

SELECT
    t_final.PANNumber
    , t_final.[Transaction Serial No]
    , t_final.[Transaction Datetime]
    , t_final.[Transaction Amount]
    , t_final.MCC
    , t_final.[Country Code]
    , t_final.[Card Acceptor Terminal ID]
    , t_final.[Card Acceptor Name]
    , t_final.[Card Acceptor City]
    , t_final.[Card Acceptor Region]
    , t_final.[Card Acceptor Country Code]
    , t_final.[Currency Code]
	, tscf.[Customer Age]
	, tscf.[Branch Name]
	, tscf.[Product Code]
	, tscf.[Customer Average Income]
	, tscf.[Card Status]
	, tscf.Balance
	, tscf.[Age of Open Account Transaction]
	, tscf.[Age of Active Card Transaction]
	, tscf.[Age of Open Account Active Card]
	, tscf.[Transfer To BDI Staff]
	, tscf.[BDI Staff]
	, tscf.[Account Open Year]
	, DATEDIFF(DAY, TRY_CAST(tscf.[Card Opening Date] AS datetime), TRY_CAST(tscf.[Transaction Datetime] AS datetime)) AS 'Age of Card Open to Transaction'
	, DATEDIFF(DAY, TRY_CAST(tscf.[Registration Date E-Channel?] AS datetime), TRY_CAST(tscf.[Transaction Datetime] AS datetime)) AS 'Age of EChannel Regist to Transaction'
	, tscf.[SDB Past Due Date]
	, tscf.[Box Type Size]
	, tscf.[Box Amount]
	, tscf.[Is SDB Past Due]
	, tscf.[Flag Out Branch]
	, tscf.[Is TD Hold Amount]
	, tscf.[Is Program Hold Amount]
	, tscf.[Is Office Hour]
	, tscf.VALAS
	, tscf.[Transaction Load Time]
	, tscf.[Date Birth]
	, tscf.[Transaction Channel]
	, tscf.[Last Transaction Datetime]
	, tscf.[Last Transaction Amount]
	, tscf.[Dr Last Transaction Date]
	, tscf.[Dr Last Transaction Amount]
	, tscf.[Cr Last Transaction Date]
	, tscf.[Cr Last Transaction Amount]
	, tscf.[Total Transaction Amount Last 1 Day]
	, tscf.[Dr Total Transaction Amount Last 1 Day]
	, tscf.[Cr Total Transaction Amount Last 1 Day]
	, tscf.[Total Transaction Amount Last 7 Days]
	, tscf.[Dr Total Transaction Amount Last 7 Days]
	, tscf.[Cr Total Transaction Amount Last 7 Days]
	, tscf.[Total Transaction Amount Last 1 Month?]
	, tscf.[Dr Total Transaction Amount Last 1 Month]
	, tscf.[Cr Total Transaction Amount Last 1 Month]
	, tscf.[Average Transaction Amount Last 1 Day]
	, tscf.[Average Transaction Amount Last 7 Days]
	, tscf.[Average Transaction Amount Last 1 Month?]
	, tscf.[Total Transaction Amount L15Mi]
	, tscf.[High Risk Customer]
	, tscf.[Product Name]
	, tscf.[Previous Account Status]
	, tscf.[Remark 4 (Tabungan)]
	, tscf.HoldAmountTB
	, tscf.[Remark 4 (Deposito)]
	, tscf.[Hold Amount DP]
	, tscf.[Age of SDB Expiry Date Transaction Date]
	, tscf.[SDB Product Code]
	, tscf.[SDB Balance]
	, tscf.[Account Reactivated Date]
	, tscf.[CC Branch Code (Acc)]
	, tscf.[Branch Name (Acc)]
	, tscf.[CC Branch Code (Card)]
	, tscf.[Branch Name (Card)]
	, tscf.[CC Branch Code (Active Dormant)]
	, tscf.[Branch Name (Active Dormant)]
	, tscf.[Branch Name (Trx)]
	, tscf.[Is WhiteList?]
	, tscf.[Is Blacklist Account?]
	, tscf.[Is Blacklist Cust?]
	, tscf.[MNemonic Trx Desc]
	, tscf.[Product Code To]
	, tscf.[Access Code]
	, tscf.[SDB Box No ]
	, tscf.[SDB Status]
	, tscf.[SDB Last Payment Amount]
	, tscf.[Exclude Flag Txn Desc]
	, tscf.[Product Code (Deposito)]
	, tscf.[Is Merchant BDI?]
	, tscf.[Merchant Acc Name]
	, tscf.[Card Issuer Name]
	, tscf.[Total Trx Num L1D]
	, tscf.[Total Trx Num L7D]
	, tscf.[Is High Risk Country?]
	, tscf.[Is Blacklist Merchant?]
	, tscf.[Is WatchList Merchant?]
	, tscf.[Is High Risk MCC?]
	, tscf.[Customer Email]
	, tscf.[Credit Limit]
	, tscf.[Cash Credit Limit]
	, tscf.[Card Product]
	, tscf.[Customer No]
	, tscf.[Customer Sex]
	, tscf.[Is Blacklist Card No?]
	, tscf.[Is Valid Card?]
	, tscf.Currency
	, tscf.[POS Entry Mode]
	, tscf.[PIN Entry Capability]
	, tscf.[Is Card Expired?]
	, tscf.[Is 3DSecure?]
	, tscf.[Card Brand]
	, tscf.[Is WhiteList Merchant?]
	, tscf.[Is BDI Email?]
	, tscf.[Is WhiteList Card No?]
	, tscf.[Is WhiteList Account No?]
	, tscf.[Is Account BDI Employee?]
	, tscf.[Cod Access]
	, tscf.[Transaction Type]
	, tscf.[Total Transaction Amount L10Mi]
	, tscf.[Customer Type]
	, tscf.[Total Transaction Amount L5min]
	, tscf.TotalTrxAmountContactless
	, tscf.[Branch City]
	, tscf.[Branch Zip Code]
	, tscf.[Is Whitelist CardNo Open?]
	, tscf.DaysOfWhitelistCardNoOpen
	, tscf.[IsVAAccount?]
	, tscf.[Is Exclude Teminal Id?]
	, tscf.isBDICard
	, tscf.[Is Watchlist Card No?]
	, tscf.[Source Channel]
	, tscf.IsBlackListTID
	, tscf.IsFirstTrx3DS
	, tscf.[Card Data Input Capability]
	, tscf.CustomerNationality
	, tscf.[Total Transaction Amount Contactless per EOD]
	, tscf.IsTopUp
	, tscf.IsBillPayment
	, tscf.IsPaymentToVA
	, tscf.[Total Transaction Amount to Same EWallet L1D]
	, tscf.[Total Transaction Amount to Diff Wallet L1D]
	, tscf.[Total Transaction Amount to Same VA L1D]
	, tscf.[Total Transaction Amount to Diff VA L1D]
	, tscf.[Is High Risk Country Threshold Amount?]
	, tscf.[Is High Risk Currency?]
	, tscf.[Is High Risk Currency Threshold Amt?]
	, tscf.IsQRtrx
	, tscf.IsDCashTrx
	, tscf.IsFTTrx
	, tscf.IsDCashTrx1
	, tscf.IsFTTrx1
	, tscf.[CummTrxAmountQRIS 10Mins]
	, tscf.[CummTrxAmountDcash 15Mins]
	, tscf.[CummAmount Trf Last 10min]
	, tscf.[TrxQRIS Last 7D]
	, tscf.[TransactionTrf Last 7D]
	, tscf.[CummAmount Trf Last 20min]
	, tscf.[CountTrxQRIS2 10mins]
	, tscf.[CountTrxDcash 15Mins]
	, tscf.[CountTrxTrf2 10Mins]
	, tscf.[CountTrxVA Last 7D]
	, tscf.[CountTrxInqVA 15mins]
	, tscf.[Total Trx Num L15Mins]
	, tscf.[TOTAL AMOUNT LESS THAN 2 DAYS]
	, tscf.[Is Whitelist CardNo Previlage ?]
	, tscf.IsWhiteListTID
	, tscf.IsWatchListTID
	, tscf.[Time Difference between Current and Previous Transaction]
	, tscf.[Total Transaction Amount L30Mi]
	, tscf.IsBucket1WatchListCardNo
	, tscf.IsBucket1BlacklistCardNo
	, tscf.IsBucket1WhiteListCardNo
	, tscf.IsBucket2WhiteListMerchant
	, tscf.IsBucket2WatchListMerchant
	, tscf.IsBucket2BlacklistMerchant
	, tscf.CountTrx11PMto2AM
	, tscf.IsBucket3WhiteListMerchant
	, tscf.IsBucket3WatchListMerchant
	, tscf.IsBucket3WhiteListMID
	, tscf.IsBucket3WatchListMID
	, tscf.IsBucket3BlacklistMID
	, tscf.IsFacebookBlacklistMerchant
	, t_final.Confirmed
FROM t_final
LEFT JOIN Transaction_Summary_Calculations_Fraud tscf
	ON t_final.[Transaction Serial No] = tscf.[Transaction Serial No]
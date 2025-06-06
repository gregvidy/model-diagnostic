/*
	2025-05-16 v0.2 Adam
*/

DECLARE @ExtractDate DATETIME
SET @ExtractDate = GETDATE();

SELECT
	Transaction_Serial_No AS 'Transaction_Serial_No',
	MTI AS 'MTI',
	PrimaryBitMap AS 'PrimaryBitMap',
	SecondaryBitMap AS 'SecondaryBitMap',
	HashBytes('SHA2_256', PANNumber) AS 'PANNumber',
	ProcessingCode AS 'ProcessingCode',
	TransactionAmount AS 'TransactionAmount',
	TransamisionDateTime AS 'TransamisionDateTime',
	STAN AS 'STAN',
	Transaction_Datetime AS 'Transaction Datetime',
	SetllementDate AS 'SetllementDate',
	CaptureDate AS 'CaptureDate',
	MCC AS 'MCC',
	CardDataInputCapability AS 'Terminal Capability',
	CardholderAuthCapability AS 'CardholderAuthCapability',
	CardCaptureCapability AS 'CardCaptureCapability',
	OS AS 'OS',
	CardholderPresent AS 'CardholderPresent',
	CardPresent AS 'CardPresent',
	CardInputMode AS 'POS Entry Mode',
	CardholderAuthMethod AS 'CardholderAuthMethod',
	CardholderAuthEntity AS 'CardholderAuthEntity',
	CardDataOutputCapability AS 'CardDataOutputCapability',
	TerminalOutputCapability AS 'TerminalOutputCapability',
	PINCaptureCapability AS 'PINCaptureCapability',
	AcquiringID AS 'AcquiringID',
	ForwardingICC AS 'ForwardingICC',
	RRN AS 'RRN',
	CardAcceptorTerminalID AS 'CardAcceptorTerminalID',
	HashBytes('SHA2_256', CardAcceptorName) AS 'CardAcceptorName',
	HashBytes('SHA2_256', CardAcceptorStreet) AS 'CardAcceptorStreet',
	CardAcceptorCity AS 'CardAcceptorCity',
	CardAcceptorPostalCode AS 'CardAcceptorPostalCode',
	CardAcceptorRegionCode AS 'CardAcceptorRegionCode',
	CardAcceptorCountryCode AS 'CardAcceptorCountryCode',
	AdditionalData AS 'Additional Data',
	TransactionCurrencyCode AS 'TransactionCurrencyCode',
	OriginalMessageType AS 'OriginalMessageType',
	OriginalSTAN AS 'OriginalSTAN',
	OriginalTransmissionDatetime AS 'OriginalTransmissionDatetime',
	HashBytes('SHA2_256', OriginalAcquiringID) AS 'OriginalAcquiringID',
	BillCompanyID AS 'BillCompanyID',
	BIllNumber AS 'BIllNumber',
	HashBytes('SHA2_256', BillConsumerNumber) AS 'BillConsumerNumber',
	BillRefNo1 AS 'BillRefNo1',
	BillRefNo2 AS 'BillRefNo2',
	HashBytes('SHA2_256', AccountIdentification1) AS 'Debit Account No',
	HashBytes('SHA2_256', AccountIdentification2) AS 'Credit Account No',
	ORFTContraBankID AS 'ORFTContraBankID',
	HashBytes('SHA2_256', ORFTContraBankAccNo) AS 'ORFTContraBankAccNo',
	Preauthhold AS 'Preauthhold',
	PreAuthSeqNumber AS 'PreAuthSeqNumber',
	HashBytes('SHA2_256', ReferalPhoneNumber) AS 'ReferalPhoneNumber',
	HashBytes('SHA2_256', MemberNumber) AS 'MemberNumber',
	POSTerminalID AS 'POSTerminalID',
	SICCode AS 'SICCode'
INTO [DataExtractionDB].[dbo].[C06_Details_Hashed]
FROM [BDI-PRD-PDTORDB].[Predator].[dbo].[C06_Details] WITH (NOLOCK) WHERE Transaction_Datetime <= @ExtractDate
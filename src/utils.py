import re
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import List, Tuple
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def remove_highly_correlated_features(df, threshold, numerical_cols, method="pearson", verbose=True):
    """
    Remove features that are highly correlated with each other
    """
    df_numeric = df[numerical_cols]
    corr_matrix = df_numeric.corr(method=method).abs()

    # upper triangle mask
    upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    upper_corr = corr_matrix.where(upper)

    to_drop = [col for col in upper_corr.columns if any(upper_corr[col] > threshold)]
    if verbose:
        print(f"Removed columns: {to_drop}")

    df_reduced = df.drop(columns=to_drop)
    return df_reduced, to_drop
    

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
):
    y_pred = (y_prob >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    
    # plot using seaborn heatmap
    tn, fp, fn, tp = cm.ravel()
    labels = ["Non-Fraud", "Fraud"]
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
    print(f"Recall/True Positive Rate: The model got {np.round(tp/(tp+fn)*100, 2)}% of correctly predicted fraud from all of the actual fraud transactions")
    print(f"Precision: The model got {np.round(tp/(tp+fp)*100, 2)}% of correctly predicted fraud from all transactions that predicted as fraud")
    print(f"False Positive Rate: The model got {np.round(fp/(fp+tn)*100, 2)}% of incorrectly predicted a fraud transactions when the actual transactions is non-fraud")


def generate_pred_df(
    X,
    y,
    clf,
    amount_col='Transaction Amount',
    bin_on='pbad',
    increment_bin_rate=0.025,
    bins=None
):
    """
    Generate prediction dataframe with all necessary columns for bin analysis
    """
    if bins is None:
        bins = np.arange(0, (1+increment_bin_rate), increment_bin_rate)

    # predict probabilities once
    proba = clf.predict_proba(X)
    pbad = proba[:, 1]
    pgood = proba[:, 0]

    total_trnx_amt = X[amount_col]
    total_bad_trnx_amt = np.where(y==1, total_trnx_amt, 0)
    total_good_trnx_amt = np.where(y==0, total_trnx_amt, 0)

    # binning based on user selection
    bin_target = pbad if bin_on == 'pbad' else pgood
    bin_column = pd.cut(bin_target, bins=bins,
                        include_lowest=True, right=False)

    # build dataframe
    pred_df = pd.DataFrame({
        "trnx_id": X.index,
        "pbad": pbad,
        "pgood": pgood,
        "bin": bin_column,
        "is_bad": y,
        "is_good": 1-y,
        "total_trnx_amt": total_trnx_amt,
        "total_bad_trnx_amt": total_bad_trnx_amt,
        "total_good_trnx_amt": total_good_trnx_amt,
        "total_expected_loss_amt": np.round((pbad * total_trnx_amt), 2)
    })

    return pred_df


def compute_bin_aggregates(pred_df):
    """
    Compute aggregation metrics for scoring bin analysis
    """
    agg_cols = {
        "trnx_id": "count",
        "is_bad": "sum",
        "is_good": "sum",
        "total_trnx_amt": "sum",
        "total_bad_trnx_amt": "sum",
        "total_good_trnx_amt": "sum",
        "total_expected_loss_amt": "sum",
    }

    aggregated_df = pred_df.groupby("bin").agg(agg_cols)
    total_records = aggregated_df["trnx_id"].sum()

    # compute standard rates
    aggregated_df["bad_rate"] = aggregated_df["is_bad"] / aggregated_df["trnx_id"]
    aggregated_df["good_rate"] = aggregated_df["is_good"] / aggregated_df["trnx_id"]

    # reverse cumulative helper
    def reverse_cumsum(series):
        return series[::-1].cumsum()[::-1]

    aggregated_df["cum_records_passed"] = reverse_cumsum(aggregated_df["trnx_id"])
    aggregated_df["cum_records_rejected"] = aggregated_df["trnx_id"].cumsum()
    aggregated_df["cum_good"] = reverse_cumsum(aggregated_df["is_good"])
    aggregated_df["cum_bad"] = reverse_cumsum(aggregated_df["is_bad"])

    aggregated_df["pass_through_rate"] = aggregated_df["cum_records_passed"] / total_records
    aggregated_df["pass_through_rejected"] = aggregated_df["cum_records_rejected"] / total_records
    aggregated_df["cum_good_rate"] = aggregated_df["cum_good"] / aggregated_df["cum_records_passed"]
    aggregated_df["cum_bad_rate"] = aggregated_df["cum_bad"] / aggregated_df["cum_records_passed"]

    aggregated_df["cum_total_trnx_amt"] = reverse_cumsum(aggregated_df["total_trnx_amt"])
    aggregated_df["cum_bad_total_trnx_amt"] = reverse_cumsum(aggregated_df["total_bad_trnx_amt"])
    aggregated_df["cum_good_total_trnx_amt"] = reverse_cumsum(aggregated_df["total_good_trnx_amt"])
    aggregated_df["cum_total_expected_loss_amt"] = reverse_cumsum(aggregated_df["total_expected_loss_amt"])

    return aggregated_df


def get_features_by_missing_pct(
    df: pd.DataFrame,
    threshold: float,
    exclude_cols: List[str] == None,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Get features with missing value percentage 
    """
    # calculate percentage of missing value
    missing_pct = df.isna().mean()

    # build summary df
    summary_df = pd.DataFrame({
        'feature': missing_pct.index,
        'missing_pct': missing_pct.values
    })

    # filter out excluded columns before applying threshold
    filtered_summary = summary_df[
        ~summary_df['feature'].isin(exclude_cols)
    ]

    # select columns meeting the threshold
    selected_cols = filtered_summary[
        filtered_summary['missing_pct'] < threshold
    ]['feature'].tolist()

    return selected_cols, summary_df

    
def read_query_file(query_path: Path):
    """
    Read SQL file from query path, then return it as a string
    """
    with open(query_path, "r", encoding="utf-8") as file:
        query = file.read()
        return query


def extract_other_product(description):
    # Define pattern
    product_patterns = {
        r"traveloka3ds": "traveloka",
        r"adv parking": "adv parking",
        r"xanh sm": "xanh sm",
        r"hotelcom": "hotel.com",
        r"7-eleven": "7-eleven",
        r"transnusa": "transnusa",
        r"lazada": "lazada",
        r"uber": "uber",
        r"rwgenting": "resort world genting",
        r"playerstech": "playerstech",
        r"ayam berjaya": "ayam berjaya",
        r"1password": "1password",
        r"jetbrains": "jet brains",
        r"mandira travel": "mandira travel",
        r"homecenterid": "home center id",
        r"tokopedia": "tokopedia",
        r"starlink": "starlink",
        r"facebk": "facebook",
        r"bookingcom": "booking.com",
        r"booking.com": "booking.com",
        r"javamifi": "javamifi",
        r"webhost": "web host",
        r"science22com": "science22.com",
        r"doordash": "doordash",
    }

    # Search for patterns in the description
    for pattern, product in product_patterns.items():
        if re.search(pattern, description):
            return product

    # If no pattern matches, return the original description in lowercase
    return description


def extract_google_product(description):
    # Define patterns for Google products
    product_patterns = {
        r"googleads": "google ads",
        r"googlecloud": "google cloud",
        r"googleone": "google one",
        r"googleplay": "google play",
        r"googlemaps": "google maps",
        r"googleworkspace": "google workspace",
        r"googleanalytics": "google analytics",
        r"googledrive": "google drive",
        r"googlemeet": "google meet",
        r"googlephotos": "google photos",
        r"googlepay": "google pay",
        r"googlefiber": "google fiber",
        r"googlewifi": "google wifi",
        r"googlehome": "google home",
        r"googleassistant": "google assistant",
        r"googlefit": "google fit",
        r"googleduo": "google duo",
        r"googlevoice": "google voice",
        r"googleclassroom": "google classroom",
        r"googleearth": "google earth",
        r"googlefinance": "google finance",
        r"googlekeep": "google keep",
        r"googletranslate": "google translate",
        r"googletrends": "google trends",
        r"googleadsense": "google adsense",
        r"googleadwords": "google adwords",
        r"googlemerchant": "google merchant",
        r"googlemybusiness": "google my business",
        r"googletravel": "google travel",
        r"googleflights": "google flights",
        r"googlehotel": "google hotel",
        r"googlebooks": "google books",
        r"googlecalendar": "google calendar",
        r"googlecontacts": "google contacts",
        r"googlecurrents": "google currents",
        r"googlemessages": "google messages",
        r"googlenews": "google news",
        r"googlenow": "google now",
        r"googlesearch": "google search",
        r"googlesheets": "google sheets",
        r"googleslides": "google slides",
        r"googlesites": "google sites",
        r"googletasks": "google tasks",
        r"googletv": "google tv",
        r"googlewallet": "google wallet",
        r"googleweather": "google weather",
        r"googlewellbeing": "google wellbeing",
        r"googleyoutube": "google youtube",
    }

    # Search for patterns in the description
    for pattern, product in product_patterns.items():
        if re.search(pattern, description):
            return product

    # If no pattern matches, return the original description in lowercase
    return description


def extract_provider_name(description):
    # Step 1a: Convert to lowercase
    description = description.lower() if description is not None else 'NA'

    # Step 1b: Directly extract product name from raw
    description = extract_google_product(description)
    description = extract_other_product(description)

    # Split into words
    words = description.split()

    if all(word.isalpha() for word in words):
        description = extract_google_product(description)
        return description.strip()
    else:
        # Step 2a: extract google products
        description = extract_google_product(description)
        # Step 2b: Remove text after special characters (*, -, _)
        description = re.split(r"[\*\-_]", description)[0]

        # Step 3: Extract domain name (remove 'www.')
        domain_pattern = r"\b(?:www\.)?([a-zA-Z0-9\-]+\.[a-zA-Z]{2,})\b"
        domain_match = re.search(domain_pattern, description)
        if domain_match:
            return domain_match.group(1)

        # Step 4: Remove combined alphanumeric-character words
        description = re.sub(r"\b[a-zA-Z0-9]+-[a-zA-Z0-9-]+\b", "", description)

        # Step 5: Remove alphanumeric words (likely IDs)
        description = re.sub(r"\b[a-zA-Z0-9]*\d+[a-zA-Z0-9]*\b", "", description)

        # Step 6: Extract first clean alphabetic word
        matches = re.findall(r"\b[a-zA-Z]+\b", description)
        return matches[0] if matches else description.strip()


def categorize_terminal_owner(owner):
    owner = str(owner).lower()
    if any(
        keyword in owner
        for keyword in [
            "alibaba",
            "distro",
            "hellofresh",
            "amazon",
            "taobao",
            "amzn",
            "ebay",
            "shop",
            "store",
            "market",
            "commerce",
            "tokopedia",
            "shopee",
            "blibli",
            "lazada",
            "zalora",
            "temu",
            "aliexpress",
            "bukalapak",
        ]
    ):
        return "ecommerce"
    elif any(
        keyword in owner
        for keyword in [
            "oculus",
            "netflix",
            "spotify",
            "youtube",
            "digital",
            "stream",
            "media",
            "apple music",
            "disney+",
            "viu",
            "iflix",
            "crunchyroll",
            "scribd",
            "kindle",
        ]
    ):
        return "digital products"
    elif any(
        keyword in owner
        for keyword in [
            "bloomberg",
            "nba",
            "ufc",
            "canva",
            "samsung",
            "zoom",
            "discord",
            "bitly",
            "wordpress",
            "classpass",
        ]
    ):
        return "online subscription"
    elif any(keyword in owner for keyword in ["ticket", "reverbnation"]):
        return "tickets"
    elif any(
        keyword in owner
        for keyword in [
            "starlink",
            "finpay",
            "dana",
            "telkomsel",
            "indosat",
            "axiata",
            "hutchinson",
            "pln",
            "xl axiata",
            "smartfren",
            "sp powerpacsg",
        ]
    ):
        return "utilities"
    elif any(
        keyword in owner
        for keyword in [
            "digimap",
            "erafone",
            "istore",
            "ibox",
            "electronic city",
            "best denki",
        ]
    ):
        return "retail - electronics"
    elif any(
        keyword in owner
        for keyword in ["ikea", "ace hardware", "informa", "depo bangunan"]
    ):
        return "retail - furniture"
    elif any(keyword in owner for keyword in ["mall"]):
        return "retail - others"
    elif any(
        keyword in owner
        for keyword in [
            "udemy",
            "coursera",
            "edx",
            "learn",
            "education",
            "ruang guru",
            "zenius",
            "skillshare",
            "duolingo",
            "khan academy",
        ]
    ):
        return "e-learning"
    elif any(
        keyword in owner
        for keyword in [
            "garuda",
            "singaporeair",
            "sq",
            "airbnb",
            "booking",
            "travel",
            "trip",
            "tour",
            "agoda",
            "expedia",
            "tiket.com",
            "traveloka",
            "trip.com",
        ]
    ):
        return "travel"
    elif any(
        keyword in owner
        for keyword in [
            "careem",
            "bolt",
            "uber",
            "lyft",
            "grab",
            "ride",
            "taxi",
            "blue bird",
            "xanh",
            "gojek",
            "maxim",
            "indrive",
            "beam",
            "lime",
        ]
    ):
        return "transportation"
    elif any(
        keyword in owner
        for keyword in [
            "apple",
            "microsoft",
            "google",
            "tech",
            "software",
            "openai",
            "midjourney",
            "ai",
            "dropbox",
            "adobe",
            "notion",
            "slack",
            "github",
        ]
    ):
        return "technology"
    elif any(
        keyword in owner
        for keyword in [
            "domino",
            "coffee",
            "doordash",
            "mcdonalds",
            "starbucks",
            "restaurant",
            "food",
            "beverage",
            "kfc",
            "dominos",
            "grabfood",
            "gofood",
            "foodpanda",
        ]
    ):
        return "food & beverage"
    elif any(
        keyword in owner
        for keyword in [
            "blizzard",
            "sony",
            "nintendo",
            "playstation",
            "xbox",
            "game",
            "gaming",
            "steam",
            "garena",
            "mobile legends",
            "pubg",
            "roblox",
            "genshin",
            "riot games",
            "activation",
        ]
    ):
        return "gaming"
    elif any(
        keyword in owner
        for keyword in [
            "facebook",
            "twitter",
            "instagram",
            "social",
            "media",
            "tiktok",
            "linktree",
            "heylink",
            "onlyfans",
        ]
    ):
        return "social media"
    elif any(
        keyword in owner
        for keyword in [
            "paper.id",
            "pay",
            "doku",
            "bank",
            "paypal",
            "finance",
            "financial",
            "payoneer",
            "stripe",
            "wise",
            "dana",
            "ovo",
            "gopay",
            "linkaja",
            "bdi",
            "bca",
            "danamon",
            "mandiri",
            "bni",
            "bjb",
            "bri",
        ]
    ):
        return "finance"
    elif any(
        keyword in owner
        for keyword in [
            "dentist",
            "health",
            "wellness",
            "fitness",
            "gym",
            "mindvalley",
            "celebrityfitness",
            "strong marriage now",
            "betterhelp",
        ]
    ):
        return "health & wellness"
    elif any(
        keyword in owner
        for keyword in [
            "clinic",
            "hospital",
            "insurance",
            "bpjs",
            "prudential",
            "allianz",
            "axa",
        ]
    ):
        return "medical & insurance"
    elif any(
        keyword in owner
        for keyword in [
            "immigration",
            "tax",
            "police",
            "npwp",
            "bpjs",
            "lta",
            "sp powerpacsg",
        ]
    ):
        return "government & public services"
    elif any(
        keyword in owner
        for keyword in [
            "hotel",
            "resort",
            "inn",
            "motel",
            "hostel",
            "lodging",
            "guesthouse",
            "bnb",
            "marriott",
            "hilton",
            "hyatt",
            "accor",
            "pullman",
            "citadines",
            "fairfield",
            "novotel",
            "ritz",
            "intercontinental",
            "holiday inn",
            "reddoorz",
            "airy",
            "oyo",
            "aston",
            "santika",
            "pop hotel",
            "ibis",
            "amaris",
            "favehotel",
            "luxehotel",
            "mandarin oriental",
            "shangri-la",
            "four seasons",
            "capella",
            "the fullerton",
            "raffles hotel",
            "kempinski",
        ]
    ):
        return "hotels & resorts"
    elif any(
        keyword in owner
        for keyword in [
            "petrol",
            "gas",
            "fuel",
            "shell",
            "pertamina",
            "esso",
            "caltex",
            "total",
            "bp",
            "petronas",
            "mobil",
            "gas station",
            "spbu",
            "bensinstation",
            "refuel",
            "refueling",
        ]
    ):
        return "petrol & gas stations"
    else:
        return "other"


def clean_categorize_merchant_name(description):
    provider = extract_provider_name(description)
    return categorize_terminal_owner(provider)
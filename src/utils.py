import re
from pathlib import Path


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
    description = description.lower()

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

from google_play_scraper import Sort, reviews
import json
from datetime import datetime


def convert_datetime(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

def fetch_app_reviews(app_list, reviews_per_rating=10, total_reviews=500):
    all_app_reviews = {}

    for app in app_list:
        print(f"\nFetching reviews for {app['name']}...")
        result, _ = reviews(
            app['package_name'],
            lang='en',
            country='us',
            sort=Sort.NEWEST,
            count=total_reviews,
            filter_score_with=None
        )

        reviews_by_rating = {1: [], 2: [], 3: [], 4: [], 5: []}
        for review in result:
            score = review['score']
            if len(reviews_by_rating[score]) < reviews_per_rating:
                reviews_by_rating[score].append({
                    'content': review['content'],
                    'score': score,
                    'date': review['at']
                })

        # Check if we have enough reviews for each rating
        if all(len(reviews) >= reviews_per_rating for reviews in reviews_by_rating.values()):
            all_app_reviews[app['name']] = reviews_by_rating
        else:
            print(f"Skipping {app['name']} due to insufficient reviews across all ratings.")

    return all_app_reviews

# List of apps to fetch reviews for
apps = [
    # Food & Drink
    {"name": "DoorDash", "package_name": "com.dd.doordash"},
    {"name": "McDonald's", "package_name": "com.mcdonalds.app"},
    {"name": "Uber Eats", "package_name": "com.ubercab.eats"},
    
    # Popular Apps
    {"name": "TikTok", "package_name": "com.zhiliaoapp.musically"},
    {"name": "Instagram", "package_name": "com.instagram.android"},
    {"name": "Facebook", "package_name": "com.facebook.katana"},
    
    # Business Tools
    {"name": "Microsoft Office", "package_name": "com.microsoft.office.officehubrow"},
    {"name": "Slack", "package_name": "com.Slack"},
    {"name": "Trello", "package_name": "com.trello"},
    
    # Video Editors & Players
    {"name": "Adobe Premiere Rush", "package_name": "com.adobe.premiererush.videoeditor"},
    {"name": "KineMaster", "package_name": "com.nexstreaming.app.kinemasterfree"},
    {"name": "VLC", "package_name": "org.videolan.vlc"},
    
    # Health & Fitness
    {"name": "MyFitnessPal", "package_name": "com.myfitnesspal.android"},
    {"name": "Strava", "package_name": "com.strava"},
    {"name": "Calm", "package_name": "com.calm.android"},
    
    # Educational Apps
    {"name": "Duolingo", "package_name": "com.duolingo"},
    {"name": "Khan Academy", "package_name": "org.khanacademy.android"},
    {"name": "Quizlet", "package_name": "com.quizlet.quizletandroid"},
    
    # Music & Audio
    {"name": "Spotify", "package_name": "com.spotify.music"},
    {"name": "SoundCloud", "package_name": "com.soundcloud.android"},
    {"name": "Shazam", "package_name": "com.shazam.android"},
    
    # Art & Design
    {"name": "Canva", "package_name": "com.canva.editor"},
    {"name": "Adobe Illustrator Draw", "package_name": "com.adobe.creativeapps.draw"},
    {"name": "Pixlr", "package_name": "com.pixlr.express"},
    
    # Take Better Photos
    {"name": "VSCO", "package_name": "com.vsco.cam"},
    {"name": "Snapseed", "package_name": "com.niksoftware.snapseed"},
    {"name": "Adobe Lightroom", "package_name": "com.adobe.lrmobile"}
]

# Fetch reviews for all apps
all_reviews = fetch_app_reviews(apps)

# Save reviews to a JSON file
with open('diverse_app_reviews.json', 'w', encoding='utf-8') as f:
    json.dump(all_reviews, f, ensure_ascii=False, indent=4, default=convert_datetime)

print("Reviews have been saved to 'diverse_app_reviews.json'")

# Display sample reviews for each app and rating
for app_name, app_reviews in all_reviews.items():
    print(f"\n=== Reviews for {app_name} ===")
    for rating in range(1, 6):
        print(f"\nRating {rating} star reviews:")
        for review in app_reviews[rating][:3]:  # Display only 3 reviews per rating
            print(f"Content: {review['content']}")
            print(f"Score: {review['score']}")
            print(f"Date: {review['date']}")
            print("---")

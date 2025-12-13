from steam_scraper import DataFetcher
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-u', '--update') # set to false to skip existing games in db
    parser.add_argument('-r', '--refresh') # refresh ranking

    fetcher = DataFetcher(
        db_path="steam_top_100x200.db",
        target_count=100,
        filter_mode="globaltopsellers",
    )

    # set 
    args = parser.parse_args()
    update = True if args.update == 'y' else False
    refresh = True if args.refresh == 'y' else False

    if update:
        fetcher.fetch_and_store()

    # Round 2: Use stored URL to scrape details
    fetcher.fetch_additional_details_from_store(update=update)

    # Round 3: Scrape top 100 reviews for each game
    fetcher.scrape_reviews(reviews_per_game=200, update=update)

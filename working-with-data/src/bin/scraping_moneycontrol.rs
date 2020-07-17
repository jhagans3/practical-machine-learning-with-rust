use reqwest;
use scraper::{Html, Selector};
use std::time::{SystemTime, UNIX_EPOCH};

// cargo run --bin scraping
#[tokio::main]
async fn main() -> Result<(), reqwest::Error> {
    let start = SystemTime::now();
    let since_the_epoch = start
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");

    let resp = reqwest::get(
        "https://www.moneycontrol.com/india/stockpricequote/power-generation-distribution/ntpc/NTP",
    )
    .await?;

    println!("Status: {}", resp.status());
    assert!(resp.status().is_success());

    let body = resp.text().await?;

    // println!("Body:\n\n{}", body);
    let fragment = Html::parse_document(&body);
    // site has changed?
    // let stories = Selector::parse("#Bse_Prc_tick > strong:nth-child(1)").unwrap();
    let stories =
        Selector::parse("#mktdet_2 > div:nth-child(2) > div:nth-child(3) > div.FR.gD_12").unwrap();

    for price in fragment.select(&stories) {
        let price_txt = price.text().collect::<Vec<_>>();
        if price_txt.len() == 1 {
            println!("{:?}", (since_the_epoch, price_txt[0]));
        }
    }

    Ok(())
}

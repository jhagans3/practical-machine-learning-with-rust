use serde::{de, Deserialize, Deserializer, Serialize};
use serde_json::Result;
use std::error::Error;
use std::fs::File;
use std::str::FromStr;
use std::vec::Vec;

pub fn untyped_example() -> Result<()> {
    let the_file = r#"{
        "FirstName": "John",
        "LastName": "Doe",
        "Age": 43,
        "Address": {
            "Street": "Downing Street 10",
            "City": "London",
            "Country": "Great Britain"
        },
        "PhoneNumbers": [
            "+44 1234567",
            "+44 2345678"
        ]
    }"#;
    // random json string
    let person: serde_json::Value =
        serde_json::from_str(the_file).expect("JSON was not well-formatted");
    let address = person.get("Address").unwrap();
    println!("{:?}", address.get("City").unwrap());

    Ok(())
}

#[derive(Serialize, Deserialize)]
#[allow(non_snake_case)]
struct Person {
    FirstName: String,
    LastName: String,
    Age: u8,
    Address: Address,
    PhoneNumbers: Vec<String>,
}

#[derive(Serialize, Deserialize)]
#[allow(non_snake_case)]
struct Address {
    Street: String,
    City: String,
    Country: String,
}

fn typed_example() -> Result<()> {
    // Some JSON input data as a &str. Maybe this comes from the user.
    let data = r#"{
        "FirstName": "John",
        "LastName": "Doe",
        "Age": 43,
        "Address": {
            "Street": "Downing Street 10",
            "City": "London",
            "Country": "Great Britain"
        },
        "PhoneNumbers": [
            "+44 1234567",
            "+44 2345678"
        ]
    }"#;

    // Parse the string of data into a Person object. This is exactly the
    // same function as the one that produced serde_json::Value above, but
    // now we are asking it for a Person as output.
    let p: Person = serde_json::from_str(data)?;

    // Do things just like with any other Rust data structure.
    println!(
        "Please call {} at the number {}",
        p.FirstName, p.PhoneNumbers[0]
    );

    Ok(())
}

#[derive(Debug, Serialize, Deserialize)]
struct Prizes {
    prizes: Vec<Prize>,
}

#[derive(Debug, Serialize, Deserialize)]
#[allow(non_snake_case)]
struct Prize {
    category: String,
    #[serde(default)]
    overallMotivation: Option<String>,
    laureates: Vec<NobelLaureate>,
    #[serde(deserialize_with = "de_u16_from_str")]
    year: u16,
}

#[derive(Debug, Serialize, Deserialize)]
struct NobelLaureate {
    share: String,
    #[serde(default)]
    motivation: Option<String>,
    surname: String,
    #[serde(deserialize_with = "de_u16_from_str")]
    id: u16,
    firstname: String,
}
fn de_u16_from_str<'a, D>(deserializer: D) -> std::result::Result<u16, D::Error>
where
    D: Deserializer<'a>,
{
    let s = String::deserialize(deserializer)?;
    u16::from_str(&s).map_err(de::Error::custom)
}

fn nobel_prize_file_example() -> Result<()> {
    println!("from prizes json file",);
    let file = File::open("data/prize.json").expect("file should open read only");
    let prizes_data: Prizes = serde_json::from_reader(file).expect("file should be proper JSON");
    let prizes_0 = &prizes_data.prizes[0];
    println!("category: {:?}", prizes_0.category);
    println!("laureates: {:?}", prizes_0.laureates);
    println!("overall motivation: {:?}", prizes_0.overallMotivation);
    println!("year: {:?}", prizes_0.year);

    Ok(())
}

fn main() -> std::result::Result<(), Box<dyn Error>> {
    untyped_example()?;

    typed_example()?;

    nobel_prize_file_example()?;

    Ok(())
}

#![warn(clippy::all)]

use std::fs;
use std::path::PathBuf;
use rand::{self, Rng, seq::SliceRandom};

fn generate_image(
    image_path: &PathBuf,
    background: &image::RgbaImage,
    output_path: &PathBuf
) -> Result<(), Box<std::error::Error>> {
    let face = image::open( &image_path )?;
    let mut output = background.clone();

    let size = rand::thread_rng().gen_range( 120, 150 );
    let face = image::imageops::resize( &face, size, size, image::FilterType::Lanczos3 );
    let offset = rand::thread_rng().gen_range( 0, 256 - size );

    image::imageops::overlay( &mut output, &face, offset, 256 - size );
    let output = image::imageops::resize( &output, 224, 224, image::FilterType::Lanczos3 );
    output.save( output_path )?;

    Ok(())
}

fn main() -> Result<(), Box<std::error::Error>> {
    // Usage: cargo run --release

    // Path to directory with unpacked FERG database
    const FACES: &str = "/home/marwit/Downloads/FERG_DB_256";
    // Path to directory with backgrounds. It should contain 7 JPEGs
    // with names that are consecutive natural numbers, so
    // 1.jpg, 2.jpg, etc. where n-th image is background that should be correlated
    // with n-th emotion (see array below; 1-anger, 2-disgust, etc.)
    const BACKGROUNDS: &str = "/home/marwit/Documents/bgs";

    // Output directory
    const OUTPUT: &str = "/home/marwit/Documents/generated";
    // How many images should be generated per FACE-EMOTION pair
    const PER_EMOTE: usize = 250;

    const FACE_IDENTS: [&str; 6] = [ "aia", "bonnie", "jules", "malcolm", "mery", "ray" ];
    const EMOTION_IDENTS: [&str; 7] = [ "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise" ];

    let mut rng = rand::thread_rng();
    let mut p = 0.0;
    let backgrounds = (1 ..= EMOTION_IDENTS.len()).map( |i| {
        let mut background_path = PathBuf::from( BACKGROUNDS );
        background_path.push( format!( "{}.jpg", i ) );
        image::open( background_path ).expect( "invalid background" ).to_rgba()
    }).collect::<Vec<_>>();


    while p < 1.0 {
        println!( "ρ = {:.1}", p );

        let mut out_path = PathBuf::from( OUTPUT );
        out_path.push( format!( "{:.1}", p ) );

        if ! out_path.exists() {
            fs::create_dir( &out_path )?;
        }

        for set in [ "test", "train", "valid" ].iter() {
            for faceid in 1 ..= 7 {
                let mut ppath = out_path.clone();
                ppath.push( set );
                ppath.push( format!( "{:?}", faceid ) );
                fs::create_dir_all( ppath )?;
            }
        }

        for face in &FACE_IDENTS {
            println!( "{}", face );

            let mut path = PathBuf::from( &FACES );
            path.push( &face );

            for (i, emot) in EMOTION_IDENTS.iter().enumerate() {
                println!( "\t{}", emot );

                let mut face_path = path.clone();
                face_path.push( format!( "{}_{}", face, emot ) );

                let mut paths = vec![];

                for entry in fs::read_dir( face_path )? {
                    let entry = entry?;
                    paths.push( entry.path() );
                }

                paths.shuffle( &mut rng );

                for (k, filepath) in paths.iter().enumerate().take( PER_EMOTE ) {
                    let bgid: usize;

                    let dir = match k {
                        _ if k < (PER_EMOTE * 30) / 100 => {
                            if rng.gen::<f32>() < p {
                                bgid = i + 1;
                            } else {
                                bgid = rng.gen_range( 1, 7 );
                            }

                            "valid"
                        }
                        _ if k < PER_EMOTE / 2 => {
                            bgid = rng.gen_range( 1, 7 );

                            "test"
                        }
                        _ => {
                            if rng.gen::<f32>() < p {
                                bgid = i + 1;
                            } else {
                                bgid = rng.gen_range( 1, 7 );
                            }

                            "train"
                        }
                    };

                    let fname = filepath.file_name().unwrap();

                    let mut output_path = out_path.clone();
                    output_path.push( dir );
                    output_path.push( format!( "{}", i + 1 ) );
                    output_path.push( fname );

                    generate_image( filepath, & backgrounds[ bgid - 1 ], &output_path )?;
                }
            }
        }

        p += 0.1;
    }

    Ok(())
}

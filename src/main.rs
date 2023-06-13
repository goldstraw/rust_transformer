use rusttransformer::*;
use log::{LevelFilter};

fn main() {
    // Set the custom logger as the global logger
    log::set_logger(&logger::CustomLogger).unwrap();
    if VERBOSE {
        log::set_max_level(LevelFilter::Info);
    } else {
        log::set_max_level(LevelFilter::Error);
    }

    run::run();
}
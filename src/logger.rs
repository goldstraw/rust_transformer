use chrono::Local;
use log::{Level, Log, Metadata, Record};
use std::io::Write;

pub struct CustomLogger;

impl Log for CustomLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        // Define the log level here, e.g., Level::Info, Level::Error, etc.
        metadata.level() <= Level::Info
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            let level = record.level();
            let time = Local::now().format("%H:%M:%S");

            // Create a string containing the log message with prefix and timestamp
            let log_message = format!("[{}] {}: {}", time, level, record.args());

            // Print the log message to the console
            println!("{}", log_message);
        }
    }

    fn flush(&self) {
        // Ensure the logs are immediately written to the console
        let _ = std::io::stdout().flush();
    }
}
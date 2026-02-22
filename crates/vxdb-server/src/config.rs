use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "vxdb-server", about = "vxdb HTTP server")]
pub struct Config {
    #[arg(long, default_value = "0.0.0.0", env = "VXDB_HOST")]
    pub host: String,

    #[arg(long, default_value_t = 8080, env = "VXDB_PORT")]
    pub port: u16,
}

impl Config {
    pub fn addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

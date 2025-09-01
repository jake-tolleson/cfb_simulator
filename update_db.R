
# --- 1. Load Libraries ---
library(tidyverse)
library(lubridate)
library(cfbfastR)
library(DBI)
library(RSQLite)
library(glue) # Excellent for building SQL queries safely
library(dotenv)

setwd("/Users/jaketolleson/Library/CloudStorage/GoogleDrive-jaketolleson7@gmail.com/My Drive/01_Projects/CFB/sim_package")

# This loads the variables from the .env file in your project directory
load_dot_env()

# --- 2. Setup ---

# Define constants
DB_PATH <- Sys.getenv("db_dir")
START_YEAR <- 2014

# --- Define the true College Football Season Year ---
# A season bleeds into the next calendar year (e.g., 2025 season ends in Jan 2026)
today_date <- today()
calendar_year <- year(today_date)
current_month <- month(today_date)

if (current_month %in% c(1, 2)) {
  # If it's Jan/Feb, the active season is the PREVIOUS calendar year's season
  CURRENT_YEAR <- calendar_year - 1
} else {
  # Otherwise, the active season is the current calendar year
  CURRENT_YEAR <- calendar_year
}

cat(glue::glue("Today is {today_date}. The current CFB Season is defined as: {CURRENT_YEAR}\n\n"))

# Update PBP data first
cfbfastR::update_cfb_db(dbdir = dirname(DB_PATH),
                        dbname = basename(DB_PATH),
                        tblname = "cfbfastR_pbp",
                        force_rebuild = c(CURRENT_YEAR))


# --- 3. The "Smart Update" Function ---
# This is the core logic for updating a single table
sync_cfb_table <- function(conn, table_name, cfb_function, is_year_based = TRUE) {
  
  cat(glue::glue("\n--- Processing table: {table_name} ---\n"))
  
  # --- A. Handle non-year based tables (like venues, conferences) ---
  if (!is_year_based) {
    cat(glue::glue("Fetching complete dataset for '{table_name}'...\n"))
    Sys.sleep(6) # Stay under API rate limits
    data <- cfb_function()
    DBI::dbWriteTable(conn, table_name, as.data.frame(data), overwrite = TRUE)
    cat(glue::glue("'{table_name}' has been created/overwritten.\n"))
    return()
  }
  
  # --- B. Handle year-based tables ---
  
  # === PART 1: CATCH UP ON MISSING *HISTORICAL* SEASONS ===
  latest_completed_year <- CURRENT_YEAR - 1
  
  if (DBI::dbExistsTable(conn, table_name)) {
    # Dynamically find the year/season column
    col_names <- DBI::dbListFields(conn, table_name)
    year_column <- if ("season" %in% col_names) "season" else if ("year" %in% col_names) "year" else NULL
    
    last_year_in_db <- START_YEAR - 1
    if (!is.null(year_column)) {
      query <- glue::glue("SELECT MAX({year_column}) FROM {table_name}")
      last_year_in_db <- DBI::dbGetQuery(conn, query)[[1]]
    } else {
      warning(glue::glue("Could not find 'season' or 'year' column in '{table_name}'. Full rebuild may be necessary."))
    }
    if (is.na(last_year_in_db)) last_year_in_db <- START_YEAR - 1
    
    if (last_year_in_db < latest_completed_year) {
      years_to_pull <- (last_year_in_db + 1):latest_completed_year
      cat(glue::glue("Found missing historical seasons. Fetching: {paste(years_to_pull, collapse=', ')}\n"))
      
      historical_data <- purrr::map_dfr(years_to_pull, ~{
        cat(glue::glue("  - Fetching {table_name} for year: {.x}\n"))
        Sys.sleep(6) # Stay under API rate limits
        tryCatch({cfb_function(year = .x)}, error = function(e){cat(glue::glue("   ! Warning: {e$message}\n")); return(NULL)})
      })
      
      if (nrow(historical_data) > 0) {
        DBI::dbWriteTable(conn, table_name, as.data.frame(historical_data), append = TRUE)
        cat(glue::glue("Successfully appended {nrow(historical_data)} rows for historical seasons.\n"))
      }
    } else {
      cat("Historical seasons are all up to date.\n")
    }
  } else {
    # Table doesn't exist, so fetch everything up to last completed year
    years_to_pull <- START_YEAR:latest_completed_year
    cat(glue::glue("Table not found. Fetching all historical seasons: {paste(years_to_pull, collapse=', ')}\n"))
    
    historical_data <- purrr::map_dfr(years_to_pull, ~{
      cat(glue::glue("  - Fetching {table_name} for year: {.x}\n"))
      Sys.sleep(6) # Stay under API rate limits
      tryCatch({cfb_function(year = .x)}, error = function(e){cat(glue::glue("   ! Warning: {e$message}\n")); return(NULL)})
    })
    
    if (nrow(historical_data) > 0) {
      DBI::dbWriteTable(conn, table_name, as.data.frame(historical_data), append = FALSE) # Initial write
      cat(glue::glue("Successfully created table '{table_name}' with {nrow(historical_data)} rows.\n"))
    }
  }
  
  # === PART 2: ALWAYS REFRESH THE *CURRENT* SEASON ===
  cat(glue::glue("--> Refreshing data for the current season: {CURRENT_YEAR}\n"))
  Sys.sleep(6) # Stay under API rate limits
  current_season_data <- tryCatch({cfb_function(year = CURRENT_YEAR)}, error = function(e){cat(glue::glue("   ! Warning: {e$message}\n")); return(NULL)})
  
  if (!is.null(current_season_data) && nrow(current_season_data) > 0) {
    col_names <- colnames(current_season_data)
    year_column <- if ("season" %in% col_names) "season" else if ("year" %in% col_names) "year" else NULL
    
    if (!is.null(year_column)) {
      # Delete existing rows for the current season before appending the new ones
      delete_query <- glue::glue("DELETE FROM {table_name} WHERE {year_column} = {CURRENT_YEAR}")
      cat(glue::glue("   - Deleting existing {CURRENT_YEAR} data...\n"))
      DBI::dbExecute(conn, delete_query)
      
      # Append the freshly downloaded data
      DBI::dbWriteTable(conn, table_name, as.data.frame(current_season_data), append = TRUE)
      cat(glue::glue("   - Appended {nrow(current_season_data)} fresh rows for {CURRENT_YEAR}.\n"))
    } else {
      warning(glue::glue("Could not find 'season' or 'year' in current season data for '{table_name}'. Cannot refresh."))
    }
  } else {
    cat(glue::glue("No data available to refresh for the current season.\n"))
  }
}

# --- 4. The Main Execution Logic ---
# Define the list of all tables you want to sync
tables_to_sync <- list(
  #list(name = "team_roster", func = cfbfastR::cfbd_rosters, yearly = TRUE),
  list(name = "player_usage", func = cfbfastR::cfbd_player_usage, yearly = TRUE),
  list(name = "player_returning", func = cfbfastR::cfbd_player_returning, yearly = TRUE),
  list(name = "transfer_portal", func = cfbfastR::cfbd_recruiting_transfer_portal, yearly = TRUE),
  list(name = "draft_picks", func = cfbfastR::cfbd_draft_picks, yearly = TRUE),
  list(name = "team_recruiting_rankings", func = cfbfastR::cfbd_recruiting_team, yearly = TRUE),
  list(name = "player_recruiting_rankings", func = cfbfastR::cfbd_recruiting_player, yearly = TRUE),
  list(name = "venues", func = cfbfastR::cfbd_venues, yearly = FALSE),
  list(name = "conferences", func = cfbfastR::cfbd_conferences, yearly = FALSE),
  list(name = "team_talent", func = cfbfastR::cfbd_team_talent, yearly = TRUE)
)

# Open the database connection ONCE
con <- DBI::dbConnect(RSQLite::SQLite(), dbname = DB_PATH)

# --- Create Index on PBP table for faster queries ---
cat("\n--- Ensuring index on cfbfastR_pbp (season, week) exists... ---\n")
# The PBP table uses 'season' for the year column
index_query <- "CREATE INDEX IF NOT EXISTS pbp_season_week_idx ON cfbfastR_pbp (season, week)"
DBI::dbExecute(con, index_query)
cat("Index is in place.\n")

# Loop through the list and sync each table
for (tbl in tables_to_sync) {
  sync_cfb_table(
    conn = con, 
    table_name = tbl$name, 
    cfb_function = tbl$func, 
    is_year_based = tbl$yearly
  )
}

# --- 5. Verify and Disconnect ---
cat("\n--- Sync Complete ---\n")
cat("Final tables in the database:\n")
print(DBI::dbListTables(con))

# Close the connection when all operations are done
DBI::dbDisconnect(con)

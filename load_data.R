#### Load data ####

files <- c("raw.rds", "emo.rds", "top.rds")
files <- paste0("data/", files)
if (!file.exists(files)) {
  raw <- read.csv("data/raw_data_anon.csv")
  emo <- read.csv("data/emotions_anon.csv")
  top <- read.csv("data/topics_anon.csv")
  saveRDS(raw, files[1])
  saveRDS(emo, files[2])
  saveRDS(top, files[3])
} else {
  raw <- readRDS(files[1])
  emo <- readRDS(files[2])
  top <- readRDS(files[3])
}


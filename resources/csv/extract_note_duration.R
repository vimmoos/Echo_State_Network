require(dplyr)

# dummy csv file rn
data <- read.csv("34time13.csv")

# the textual info is saved as a numeric factor,
# save the value for the Note_on property
note_on <- levels(data$Header)[8]

notes <- data %>%
  rename(time=X0.1,
         note=X1) %>%
  filter(Header == note_on) %>%
  select(c(time, note))

notes

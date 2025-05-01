CREATE TABLE IF NOT EXISTS characters (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    character_name TEXT NOT NULL,
    character_gender TEXT,
    voice_name TEXT NOT NULL,
    voice_location TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dialogue (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    character_id INTEGER NOT NULL,
    dialogue_text TEXT NOT NULL,
    UNIQUE (character_id, dialogue_text) ON CONFLICT ABORT,
    FOREIGN KEY (character_id) REFERENCES characters(id) ON DELETE CASCADE
)
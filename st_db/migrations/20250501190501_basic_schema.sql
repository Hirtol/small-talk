CREATE TABLE IF NOT EXISTS characters (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    character_name TEXT NOT NULL,
    character_gender TEXT NOT NULL,
    voice_name TEXT NOT NULL,
    voice_location TEXT NOT NULL,

    UNIQUE(character_name, character_gender)
);

CREATE TABLE IF NOT EXISTS dialogue (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    character_id INTEGER NOT NULL,
    dialogue_text TEXT NOT NULL,
    UNIQUE (character_id, dialogue_text) ON CONFLICT ABORT,
    FOREIGN KEY (character_id) REFERENCES characters(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS voice_lines (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    dialogue_text TEXT NOT NULL,
    voice_name TEXT NOT NULL,
    voice_location TEXT NOT NULL,
    file_name TEXT NOT NULL,
    UNIQUE (dialogue_text, voice_name, voice_location) ON CONFLICT REPLACE
);

CREATE INDEX idx_voice_loc ON voice_lines (voice_name, voice_location);
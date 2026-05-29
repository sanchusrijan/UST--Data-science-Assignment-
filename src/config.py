# src/config.py

from typing import Dict, List

# Core command definitions (Commands 1-10)
CORE_COMMANDS: Dict[int, str] = {
    1: "activate do not disturb",
    2: "deactivate do not disturb",
    3: "decline the call",
    4: "pick up the call",
    5: "play the music",
    6: "pause the music",
    7: "play the next song",
    8: "play the previous song",
    9: "increase the volume",
    10: "decrease the volume"
}

# Extension command definitions (Commands 11-14)
EXTENSION_COMMANDS: Dict[int, str] = {
    11: "increase the brightness",
    12: "decrease the brightness",
    13: "start the vehicle",
    14: "stop the vehicle"
}

# All commands combined
ALL_COMMANDS: Dict[int, str] = {**CORE_COMMANDS, **EXTENSION_COMMANDS}

# Seed templates for synthetic data generation and semantic matching centroids
COMMAND_TEMPLATES: Dict[int, List[str]] = {
    1: [
        "activate do not disturb",
        "turn on do not disturb",
        "enable do not disturb",
        "put the phone on do not disturb",
        "switch on do not disturb",
        "dnd mode turn on",
        "silence all notifications",
        "activate dnd"
    ],
    2: [
        "deactivate do not disturb",
        "turn off do not disturb",
        "disable do not disturb",
        "take off do not disturb",
        "switch off do not disturb",
        "turn off dnd",
        "deactivate dnd",
        "stop silencing notifications"
    ],
    3: [
        "decline the call",
        "decline call",
        "reject the call",
        "reject incoming call",
        "don't answer the call",
        "cancel incoming call",
        "ignore the call",
        "hang up the call"
    ],
    4: [
        "pick up the call",
        "pick up call",
        "answer the call",
        "accept the call",
        "answer incoming call",
        "receive call",
        "pick call",
        "connect the call"
    ],
    5: [
        "play the music",
        "play music",
        "start the music",
        "resume the music",
        "resume music playback",
        "start playing music",
        "turn on the music",
        "play song"
    ],
    6: [
        "pause the music",
        "pause music",
        "stop the music",
        "stop playing music",
        "pause playback",
        "halt the music",
        "suspend music",
        "pause song"
    ],
    7: [
        "play the next song",
        "play next song",
        "skip this song",
        "skip song",
        "next song",
        "go to next track",
        "next track please",
        "skip to next song"
    ],
    8: [
        "play the previous song",
        "play previous song",
        "go back a song",
        "previous song",
        "go to previous track",
        "replay the last song",
        "previous track please",
        "go back to previous track"
    ],
    9: [
        "increase the volume",
        "increase volume",
        "volume up",
        "make it louder",
        "raise the volume",
        "turn up the volume",
        "turn the volume up",
        "louder please"
    ],
    10: [
        "decrease the volume",
        "decrease volume",
        "volume down",
        "make it quieter",
        "lower the volume",
        "turn down the volume",
        "turn the volume down",
        "softer please"
    ],
    # Extension set
    11: [
        "increase the brightness",
        "increase brightness",
        "brightness up",
        "make the screen brighter",
        "raise the brightness",
        "turn up brightness",
        "brighten the screen",
        "brighter please"
    ],
    12: [
        "decrease the brightness",
        "decrease brightness",
        "brightness down",
        "make the screen dimmer",
        "lower the brightness",
        "turn down brightness",
        "dim the screen",
        "dimmer please"
    ],
    13: [
        "start the vehicle",
        "start the engine",
        "turn on the engine",
        "start car",
        "ignition start",
        "turn on the car",
        "start vehicle engine",
        "switch on the engine"
    ],
    14: [
        "stop the vehicle",
        "stop the engine",
        "turn off the engine",
        "stop car",
        "kill the engine",
        "turn off the car",
        "shut down the engine",
        "switch off the car"
    ]
}

# Standard Out-of-Scope (OOS) queries for testing rejection
OOS_TEMPLATES: List[str] = [
    "what is the weather today",
    "how is the traffic on my route",
    "navigate to the nearest petrol station",
    "give me directions to the office",
    "what time is it",
    "set a timer for ten minutes",
    "call John",
    "send a text to mom",
    "search the web for electric vehicles",
    "tell me a joke",
    "hello",
    "hi assistant",
    "who are you",
    "open the window",
    "turn off the air conditioning",
    "turn on the ac",
    "adjust seat position",
    "show me the map",
    "open the sunroof",
    "check tire pressure"
]

# Classification thresholds
# Cosine similarity threshold below which we reject as OOS
COSINE_SIMILARITY_THRESHOLD = 0.65

# TF-IDF confidence threshold (probability) below which we reject as OOS
TFIDF_CONFIDENCE_THRESHOLD = 0.45

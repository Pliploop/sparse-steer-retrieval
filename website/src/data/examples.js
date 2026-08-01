// Curated steerable-retrieval examples over music4all (MuQ-MuLan embeddings).
// Locked-in set. Real seeds (low concept alignment) + de-duped retrievals; scores are
// the steered query concept-similarity vs alpha. Labels shortened for the chip UI.
export const PLACEHOLDER = false

export const EXAMPLES = [
  {
    "id": "metal",
    "concept": "metal",
    "accent": "#7F1D1D",
    "alpha": 1.5,
    "seed": {
      "title": "Stay The Night",
      "artist": "Claptone",
      "genre": "deep house",
      "spotify": "2YL6ORlq8Ob2G5qhsvPL1M"
    },
    "results": [
      {
        "title": "Queen of the Rodeo",
        "artist": "Alice in Chains",
        "genre": "grunge",
        "affinity": 0.7352155447006226,
        "spotify": "4n6gWpXzkJzUqEdx7KroAy"
      },
      {
        "title": "The Legacy",
        "artist": "Testament",
        "genre": "thrash metal",
        "affinity": 0.7253084182739258,
        "spotify": "5xXeIlEiWIA8xnPa8BkJyj"
      },
      {
        "title": "Temple Of The Dead",
        "artist": "Candlemass",
        "genre": "doom metal",
        "affinity": 0.7184587121009827,
        "spotify": "12sI8aPJhqVSd2xW32eS7y"
      },
      {
        "title": "Elegie",
        "artist": "Patti Smith",
        "genre": "punk",
        "affinity": 0.7181934118270874,
        "spotify": "1s20rDPeaWhdDRZGVXOlb2"
      },
      {
        "title": "Angry Again",
        "artist": "Megadeth",
        "genre": "thrash metal",
        "affinity": 0.7172526121139526,
        "spotify": "3CI1JP2ooMBSFjIy1u6Yrc"
      },
      {
        "title": "Darren's Roof",
        "artist": "I Hate Myself",
        "genre": "emo",
        "affinity": 0.7159700393676758,
        "spotify": "0K6OnEm5MypRXoZFwBhYAA"
      }
    ],
    "scores": [
      -0.1938,
      -0.1526,
      -0.1092,
      -0.0644,
      -0.0189,
      0.0261,
      0.0699,
      0.1116,
      0.1508,
      0.187,
      0.2201,
      0.25,
      0.2769,
      0.3009,
      0.3224,
      0.3415,
      0.3584,
      0.3735,
      0.3869,
      0.3989,
      0.4095
    ]
  },
  {
    "id": "female-vocals",
    "concept": "female vocals",
    "accent": "#DB2777",
    "alpha": 1.5,
    "seed": {
      "title": "Winter",
      "artist": "OVERWERK",
      "genre": "electro house",
      "spotify": "4RWQMRjp08W5qkkzTCQStA"
    },
    "results": [
      {
        "title": "Eyes Without a Face",
        "artist": "Marsheaux",
        "genre": "pop",
        "affinity": 0.7315536737442017,
        "spotify": "04YObtYkElo3sNcr9hUe1w"
      },
      {
        "title": "\u0437\u0432\u0451\u0437\u0434\u044b",
        "artist": "\u041c\u044b",
        "genre": "indie pop",
        "affinity": 0.7242507934570312,
        "spotify": "2kwAK3LOqzlEedEMcRIcUN"
      }
    ],
    "scores": [
      0.0012,
      0.0346,
      0.0678,
      0.1,
      0.1306,
      0.1594,
      0.1859,
      0.21,
      0.2318,
      0.2513,
      0.2685,
      0.2838,
      0.2973,
      0.3091,
      0.3196,
      0.3288,
      0.3368,
      0.344,
      0.3503,
      0.3559,
      0.3608
    ]
  },
  {
    "id": "aggressive",
    "concept": "aggressive",
    "accent": "#DC2626",
    "alpha": 1.5,
    "seed": {
      "title": "Historias De Amor",
      "artist": "OBK",
      "genre": "pop",
      "spotify": "70iNWkA0R61bnWCin4PTbW"
    },
    "results": [
      {
        "title": "Liberate",
        "artist": "Disturbed",
        "genre": "nu metal",
        "affinity": 0.6991071105003357,
        "spotify": "2tMLPDIY5H1zICjyDLMXeT"
      },
      {
        "title": "Midnight Queen",
        "artist": "Sarc\u00f3fago",
        "genre": "thrash metal",
        "affinity": 0.6972398161888123,
        "spotify": "527SR1b4bqUhBpdcZjU9kQ"
      },
      {
        "title": "Knokkelmann",
        "artist": "Carpathian Forest",
        "genre": "black metal",
        "affinity": 0.6959702372550964,
        "spotify": "5uuVfnVLD31fc3Ezq8mgER"
      },
      {
        "title": "Endtime",
        "artist": "Katatonia",
        "genre": "doom metal",
        "affinity": 0.6906450390815735,
        "spotify": "5Tl5X02tJ1lqYqTKwWfKFA"
      },
      {
        "title": "In Ashes They Shall Reap",
        "artist": "Hatebreed",
        "genre": "hardcore",
        "affinity": 0.6897510290145874,
        "spotify": "2tTUzlFIxlyQ877NgyIPEZ"
      },
      {
        "title": "I, the Witchfinder",
        "artist": "Electric Wizard",
        "genre": "doom metal",
        "affinity": 0.6887207627296448,
        "spotify": "5Kryuu17dbt8kVN6vE61YT"
      }
    ],
    "scores": [
      -0.1179,
      -0.0823,
      -0.0454,
      -0.0084,
      0.028,
      0.0629,
      0.0958,
      0.1262,
      0.1538,
      0.1787,
      0.2009,
      0.2206,
      0.238,
      0.2534,
      0.267,
      0.2789,
      0.2895,
      0.2989,
      0.3072,
      0.3146,
      0.3212
    ]
  },
  {
    "id": "uplifting",
    "concept": "uplifting and euphoric",
    "accent": "#EC4F9E",
    "alpha": 1.5,
    "seed": {
      "title": "La marea",
      "artist": "Vetusta Morla",
      "genre": "indie rock",
      "spotify": "3ubsmZHxibRIz1q9rWf3ff"
    },
    "results": [
      {
        "title": "This Is What Makes Us Girls",
        "artist": "Lana Del Rey",
        "genre": "pop",
        "affinity": 0.7536510825157166,
        "spotify": "6QOU3FhUkPeA2RtaXcvgi7"
      },
      {
        "title": "Unluck",
        "artist": "James Blake",
        "genre": "electronic",
        "affinity": 0.7473450303077698,
        "spotify": "7wL42r11vcEhohThbDVz7Y"
      },
      {
        "title": "Go Away",
        "artist": "Katy B",
        "genre": "electronic",
        "affinity": 0.740696907043457,
        "spotify": "0m0cabE0i60F8lSeBpIAK1"
      },
      {
        "title": "Beautiful (Part ll)",
        "artist": "Wanna One",
        "genre": "k-pop",
        "affinity": 0.7317314743995667,
        "spotify": "00WGY67rwGi8IRmHn32oYZ"
      },
      {
        "title": "Die for You",
        "artist": "The Weeknd",
        "genre": "soul",
        "affinity": 0.7276137471199036,
        "spotify": "2Ch7LmS7r2Gy2kc64wv3Bz"
      },
      {
        "title": "La mia risposta",
        "artist": "Laura Pausini",
        "genre": "italian pop",
        "affinity": 0.7243170142173767,
        "spotify": "1sqmt67W6JsHjEYIhImV0i"
      }
    ],
    "scores": [
      -0.0909,
      -0.0699,
      -0.0496,
      -0.0302,
      -0.0119,
      0.0051,
      0.0207,
      0.0351,
      0.0482,
      0.0602,
      0.071,
      0.0809,
      0.0898,
      0.098,
      0.1054,
      0.1121,
      0.1183,
      0.1239,
      0.1291,
      0.1339,
      0.1383
    ]
  },
  {
    "id": "dark",
    "concept": "dark and brooding",
    "accent": "#6366F1",
    "alpha": 1.0,
    "seed": {
      "title": "Gold Star Mothers",
      "artist": "Hammock",
      "genre": "ambient",
      "spotify": "3P7b4MlExHWKsgqXTeYveQ"
    },
    "results": [
      {
        "title": "It Exists",
        "artist": "Ishome",
        "genre": "electronic",
        "affinity": 0.7608329057693481,
        "spotify": "4lJ1O5p4Zar3CHzf7DYc7R"
      },
      {
        "title": "Forgive",
        "artist": "Burial",
        "genre": "ambient",
        "affinity": 0.7493286728858948,
        "spotify": "6esTRUGkYPzGBUCvzWuZ9g"
      },
      {
        "title": "And Heaven Turned to Her Weeping",
        "artist": "John Maus",
        "genre": "dark ambient",
        "affinity": 0.7491506934165955,
        "spotify": "0YTopC5RVydOZyK7XST1iP"
      },
      {
        "title": "Thread",
        "artist": "Deaf Center",
        "genre": "dark ambient",
        "affinity": 0.7415134906768799,
        "spotify": "2P4MHjq018ktHw5ZKZXFxc"
      },
      {
        "title": "Tremendous Sea Of Love",
        "artist": "Passion Pit",
        "genre": "electropop",
        "affinity": 0.7377684712409973,
        "spotify": "3D3cEBADuPsbxRcJLAFcbY"
      },
      {
        "title": "Bornlivedie",
        "artist": "Porcupine Tree",
        "genre": "progressive rock",
        "affinity": 0.7372777462005615,
        "spotify": "3FFzvz6udyVLtlYcLTbZar"
      }
    ],
    "scores": [
      -0.1057,
      -0.0897,
      -0.0738,
      -0.0579,
      -0.0423,
      -0.027,
      -0.0122,
      0.0021,
      0.0159,
      0.029,
      0.0415,
      0.0533,
      0.0645,
      0.0751,
      0.085,
      0.0944,
      0.1031,
      0.1113,
      0.1191,
      0.1263,
      0.1331
    ]
  },
  {
    "id": "acoustic-guitar",
    "concept": "acoustic",
    "accent": "#1FA347",
    "alpha": 1.5,
    "seed": {
      "title": "methro nome",
      "artist": "Povarovo",
      "genre": "dark jazz",
      "spotify": "4CDb5OSR6x5TxEZFFAH6MR"
    },
    "results": [
      {
        "title": "C\u00e1lculos y or\u00e1culos",
        "artist": "Juana Molina",
        "genre": "folk",
        "affinity": 0.713852047920227,
        "spotify": "3NPQ6mN6QgxK1O41pGRAQO"
      },
      {
        "title": "Down",
        "artist": "Emily King",
        "genre": "soul",
        "affinity": 0.7066102027893066,
        "spotify": "5cA0vB8c9FMOVDWyJHgf26"
      },
      {
        "title": "Tuna Fish",
        "artist": "Letuce",
        "genre": "mpb",
        "affinity": 0.7040902972221375,
        "spotify": "2YVFtTLJesQHH1NZRLspdJ"
      },
      {
        "title": "Falsa Baiana",
        "artist": "Gal Costa",
        "genre": "samba",
        "affinity": 0.7011548280715942,
        "spotify": "3apeWaB90hhewS30qYZJda"
      },
      {
        "title": "Winters Kiss",
        "artist": "Blossoms",
        "genre": "rock",
        "affinity": 0.6985557675361633,
        "spotify": "4CKm5psdYjIeYftcfmzzX3"
      },
      {
        "title": "Body and Soul",
        "artist": "Tony Bennett",
        "genre": "jazz",
        "affinity": 0.693282961845398,
        "spotify": "01hJnhpAmjzg85Etnz2ECH"
      }
    ],
    "scores": [
      -0.0379,
      -0.016,
      0.005,
      0.0247,
      0.0431,
      0.06,
      0.0755,
      0.0896,
      0.1023,
      0.1139,
      0.1243,
      0.1338,
      0.1423,
      0.15,
      0.157,
      0.1633,
      0.1691,
      0.1744,
      0.1792,
      0.1836,
      0.1876
    ]
  },
  {
    "id": "epic",
    "concept": "epic and cinematic",
    "accent": "#8B5CF6",
    "alpha": 1.5,
    "seed": {
      "title": "Early Winter",
      "artist": "Gwen Stefani",
      "genre": "pop",
      "spotify": "1iHVwGKN7PKwIrj4X0erON"
    },
    "results": [
      {
        "title": "Gyllene Portarnas Bro",
        "artist": "Shining",
        "genre": "black metal",
        "affinity": 0.6692047715187073,
        "spotify": "4HpO3Gw7tXpWZgRYfaVgWD"
      },
      {
        "title": "My Last Day",
        "artist": "Gary Numan",
        "genre": "electronic",
        "affinity": 0.6551926136016846,
        "spotify": "3pwx8oyzbhD92yV33IWbm4"
      },
      {
        "title": "The Jeweller (Remastered)",
        "artist": "This Mortal Coil",
        "genre": "ambient",
        "affinity": 0.6542320251464844,
        "spotify": "3HkPayYYY6fXSVI4wuhDWT"
      },
      {
        "title": "Smokey Joe",
        "artist": "Tori Amos",
        "genre": "singer-songwriter",
        "affinity": 0.6521903872489929,
        "spotify": "0NXbSHLZoTBa0eKULxUTUe"
      },
      {
        "title": "REV 22-20 (Dry Martini Mix)",
        "artist": "Puscifer",
        "genre": "industrial",
        "affinity": 0.6513993144035339,
        "spotify": "5ibXyndDNwJyYAP3UrMBga"
      },
      {
        "title": "Curly Sue",
        "artist": "Takida",
        "genre": "rock",
        "affinity": 0.6509456634521484,
        "spotify": "6XN9uI7pnBhMUPZRdjApe2"
      }
    ],
    "scores": [
      -0.0955,
      -0.0722,
      -0.0482,
      -0.0239,
      0.0,
      0.0231,
      0.0449,
      0.0652,
      0.0839,
      0.1008,
      0.116,
      0.1296,
      0.1417,
      0.1525,
      0.1621,
      0.1707,
      0.1783,
      0.1851,
      0.1912,
      0.1966,
      0.2015
    ]
  },
  {
    "id": "dreamy",
    "concept": "dreamy and ethereal",
    "accent": "#A855F7",
    "alpha": 1.5,
    "seed": {
      "title": "If I'm In Luck I Might Get Picked Up",
      "artist": "Betty Davis",
      "genre": "funk",
      "spotify": "16gAn2l94SvrbAVgECAK19"
    },
    "results": [
      {
        "title": "Hear My Train a Comin'",
        "artist": "Jimi Hendrix",
        "genre": "classic rock",
        "affinity": 0.7096633315086365,
        "spotify": "4DmBVImaIhE3RyNvbtZTTz"
      },
      {
        "title": "Untitled (Andrea Lopez)",
        "artist": "Hype Williams",
        "genre": "lo-fi",
        "affinity": 0.702552318572998,
        "spotify": "7qmfrt2lnfVSLMVpK2ba5r"
      },
      {
        "title": "Green Mountain State",
        "artist": "Trevor Hall",
        "genre": "singer-songwriter",
        "affinity": 0.7000676989555359,
        "spotify": "0c7iF5fSBYxCuwsAv2z4iI"
      },
      {
        "title": "Dueles",
        "artist": "Jesse & Joy",
        "genre": "pop",
        "affinity": 0.6970384120941162,
        "spotify": "1iRvhKiXRElIH2Uf4gd95P"
      },
      {
        "title": "Rip X",
        "artist": "iAmJakeHill",
        "genre": "rap",
        "affinity": 0.6962800025939941,
        "spotify": "0RzAihLqsEg9Qcm4pfPbG2"
      },
      {
        "title": "Izabella",
        "artist": "Jimi Hendrix",
        "genre": "classic rock",
        "affinity": 0.693220317363739,
        "spotify": "3NSVTmtMgwTpux4pyFgQxQ"
      }
    ],
    "scores": [
      -0.1294,
      -0.103,
      -0.0755,
      -0.0477,
      -0.0202,
      0.0065,
      0.0317,
      0.0552,
      0.0767,
      0.0963,
      0.114,
      0.1298,
      0.144,
      0.1566,
      0.1678,
      0.1778,
      0.1868,
      0.1948,
      0.2019,
      0.2084,
      0.2142
    ]
  },
  {
    "id": "warm",
    "concept": "warm and intimate",
    "accent": "#B45309",
    "alpha": 1.5,
    "seed": {
      "title": "Nerves",
      "artist": "Icon for Hire",
      "genre": "alternative rock",
      "spotify": "2J7TN7bPWnzoRGjgkPykGa"
    },
    "results": [
      {
        "title": "A Matan\u00e7a do Porco",
        "artist": "Som Imagin\u00e1rio",
        "genre": "jazz",
        "affinity": 0.7101397514343262,
        "spotify": "1ojVt0t8rqPQnmF1OesdAE"
      },
      {
        "title": "Destinos",
        "artist": "Sandy e Junior",
        "genre": "pop",
        "affinity": 0.6925898194313049,
        "spotify": "3ZLjTMaK0fVXb4ktBLekwk"
      },
      {
        "title": "Destinos",
        "artist": "Sandy & Junior",
        "genre": "pop",
        "affinity": 0.6925898194313049,
        "spotify": "3ZLjTMaK0fVXb4ktBLekwk"
      },
      {
        "title": "Tomorrow I'll Be You",
        "artist": "Thursday",
        "genre": "post-hardcore",
        "affinity": 0.6823089122772217,
        "spotify": "2g5UqL21kyKLVmkcCf2mMb"
      },
      {
        "title": "Triple Cross",
        "artist": "Tiamat",
        "genre": "gothic metal",
        "affinity": 0.6809222102165222,
        "spotify": "2ZSvkKmvnQGLEa5CsB92bx"
      },
      {
        "title": "Joy in Repetition",
        "artist": "Prince",
        "genre": "soul",
        "affinity": 0.6769616007804871,
        "spotify": "5duhOzOD75R5DfjiKZtxJJ"
      }
    ],
    "scores": [
      0.0163,
      0.0367,
      0.0564,
      0.0752,
      0.0928,
      0.1091,
      0.1239,
      0.1373,
      0.1494,
      0.1602,
      0.1698,
      0.1784,
      0.186,
      0.1927,
      0.1987,
      0.204,
      0.2087,
      0.213,
      0.2167,
      0.2201,
      0.2232
    ]
  }
]

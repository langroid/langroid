// Create movie nodes
CREATE (TheMatrix:movie {title:'The Matrix', released:1999, tagline:'Welcome to the Real World'})
CREATE (TheMatrixReloaded:movie {title:'The Matrix Reloaded', released:2003, tagline:'Free your mind'})
CREATE (TheMatrixRevolutions:movie {title:'The Matrix Revolutions', released:2003, tagline:'Everything that has a beginning has an end'})
CREATE (ForrestGump:movie {title:"Forrest Gump", released:1994, tagline:"Life is like a box of chocolates..."})
CREATE (Inception:movie {title:"Inception", released:2010, tagline:"Your mind is the scene of the crime"})
CREATE (TheDarkKnight:movie {title:"The Dark Knight", released:2008, tagline:"Why So Serious?"})
CREATE (Interstellar:movie {title:"Interstellar", released:2014, tagline:"Mankind was born on Earth. It was never meant to die here."})
CREATE (PulpFiction:movie {title:"Pulp Fiction", released:1994, tagline:"Just because you are a character doesn't mean you have character."})

// Create Person nodes
CREATE (Keanu:Person {name:'Keanu Reeves', born:1964})
CREATE (Carrie:Person {name:'Carrie-Anne Moss', born:1967})
CREATE (Laurence:Person {name:'Laurence Fishburne', born:1961})
CREATE (Hugo:Person {name:'Hugo Weaving', born:1960})
CREATE (LillyW:Person {name:'Lilly Wachowski', born:1967})
CREATE (LanaW:Person {name:'Lana Wachowski', born:1965})
CREATE (JoelS:Person {name:'Joel Silver', born:1952})
CREATE (TomH:Person {name:'Tom Hanks', born:1956})
CREATE (RobertZ:Person {name:'Robert Zemeckis', born:1951})
CREATE (LeonardoD:Person {name:'Leonardo DiCaprio', born:1974})
CREATE (JosephGL:Person {name:'Joseph Gordon-Levitt', born:1981})
CREATE (EllenP:Person {name:'Ellen Page', born:1987})
CREATE (ChristopherN:Person {name:'Christopher Nolan', born:1970})
CREATE (ChristianB:Person {name:'Christian Bale', born:1974})
CREATE (HeathL:Person {name:'Heath Ledger', born:1979})
CREATE (MichaelC:Person {name:'Michael Caine', born:1933})
CREATE (MatthewM:Person {name:'Matthew McConaughey', born:1969})
CREATE (AnneH:Person {name:'Anne Hathaway', born:1982})
CREATE (JohnT:Person {name:'John Travolta', born:1954})
CREATE (UmaT:Person {name:'Uma Thurman', born:1970})
CREATE (SamuelLJ:Person {name:'Samuel L. Jackson', born:1948})
CREATE (QuentinT:Person {name:'Quentin Tarantino', born:1963})

// Create relationships for The Matrix trilogy
CREATE
(Keanu)-[:ACTED_IN {roles:['Neo']}]->(TheMatrix),
(Carrie)-[:ACTED_IN {roles:['Trinity']}]->(TheMatrix),
(Laurence)-[:ACTED_IN {roles:['Morpheus']}]->(TheMatrix),
(Hugo)-[:ACTED_IN {roles:['Agent Smith']}]->(TheMatrix),
(LillyW)-[:DIRECTED]->(TheMatrix),
(LanaW)-[:DIRECTED]->(TheMatrix),
(JoelS)-[:PRODUCED]->(TheMatrix),
(Keanu)-[:ACTED_IN {roles:['Neo']}]->(TheMatrixReloaded),
(Carrie)-[:ACTED_IN {roles:['Trinity']}]->(TheMatrixReloaded),
(Laurence)-[:ACTED_IN {roles:['Morpheus']}]->(TheMatrixReloaded),
(Hugo)-[:ACTED_IN {roles:['Agent Smith']}]->(TheMatrixReloaded),
(LillyW)-[:DIRECTED]->(TheMatrixReloaded),
(LanaW)-[:DIRECTED]->(TheMatrixReloaded),
(JoelS)-[:PRODUCED]->(TheMatrixReloaded),
(Keanu)-[:ACTED_IN {roles:['Neo']}]->(TheMatrixRevolutions),
(Carrie)-[:ACTED_IN {roles:['Trinity']}]->(TheMatrixRevolutions),
(Laurence)-[:ACTED_IN {roles:['Morpheus']}]->(TheMatrixRevolutions),
(Hugo)-[:ACTED_IN {roles:['Agent Smith']}]->(TheMatrixRevolutions),
(LillyW)-[:DIRECTED]->(TheMatrixRevolutions),
(LanaW)-[:DIRECTED]->(TheMatrixRevolutions),
(JoelS)-[:PRODUCED]->(TheMatrixRevolutions)

// Create relationships for Forrest Gump
CREATE
(TomH)-[:ACTED_IN {roles:['Forrest Gump']}]->(ForrestGump),
(RobertZ)-[:DIRECTED]->(ForrestGump)

// Create relationships for Inception
CREATE
(LeonardoD)-[:ACTED_IN {roles:['Cobb']}]->(Inception),
(JosephGL)-[:ACTED_IN {roles:['Arthur']}]->(Inception),
(EllenP)-[:ACTED_IN {roles:['Ariadne']}]->(Inception),
(ChristopherN)-[:DIRECTED]->(Inception)

// Create relationships for The Dark Knight
CREATE
(ChristianB)-[:ACTED_IN {roles:['Bruce Wayne']}]->(TheDarkKnight),
(HeathL)-[:ACTED_IN {roles:['Joker']}]->(TheDarkKnight),
(MichaelC)-[:ACTED_IN {roles:['Alfred']}]->(TheDarkKnight),
(ChristopherN)-[:DIRECTED]->(TheDarkKnight)

// Create relationships for Interstellar
CREATE
(MatthewM)-[:ACTED_IN {roles:['Cooper']}]->(Interstellar),
(AnneH)-[:ACTED_IN {roles:['Brand']}]->(Interstellar),
(MichaelC)-[:ACTED_IN {roles:['Professor Brand']}]->(Interstellar),
(ChristopherN)-[:DIRECTED]->(Interstellar)

// Create relationships for Pulp Fiction
CREATE
(JohnT)-[:ACTED_IN {roles:['Vincent Vega']}]->(PulpFiction),
(UmaT)-[:ACTED_IN {roles:['Mia Wallace']}]->(PulpFiction),
(SamuelLJ)-[:ACTED_IN {roles:['Jules Winnfield']}]->(PulpFiction),
(QuentinT)-[:DIRECTED]->(PulpFiction)

// Add some REVIEWED relationships
CREATE
(ChristopherN)-[:REVIEWED {rating: 8}]->(TheMatrix),
(QuentinT)-[:REVIEWED {rating: 9}]->(Inception),
(RobertZ)-[:REVIEWED {rating: 10}]->(TheDarkKnight),
(LeonardoD)-[:REVIEWED {rating: 9}]->(PulpFiction)
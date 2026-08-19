"""
knowledge_builder.py

Builds the RAG knowledge base (SQuAD + a curated set of hardcoded
Wikipedia-derived articles, chunked and embedded into ChromaDB). Runs
automatically on every server startup, since Render's free tier wipes
local files (including chroma_db/) on every redeploy or spin-down/spin-up
cycle.

Wikipedia coverage is hardcoded as static text (STATIC_WIKI_ARTICLES)
rather than fetched live via Wikipedia's API. Live fetching was tried
first but proved unreliable in deployment — rate limiting, User-Agent
policy rejections, and inconsistent JSON responses all caused repeated
failures. Hardcoding the content removes that entire class of problem:
no network dependency for this portion at all, guaranteed to work every
time, and just as fast as the rest of the build.
"""

import os
import shutil
import itertools

import numpy as np

import chromadb
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "knowledge_base"
CHUNK_SIZE = 100

SQUAD_ROW_LIMIT = int(os.getenv("SQUAD_ROW_LIMIT", "5000"))

# Curated Wikipedia topics fetched directly by title via Wikipedia's API,
# instead of streaming through a random slice of the dump and hoping the
# right articles happen to be included. This is both faster (a couple
# dozen direct API calls vs. streaming thousands of dump rows) and more
# reliable — it guarantees coverage of common general-knowledge questions
# (like "What is AI?") rather than leaving it to chance.
# Hardcoded static reference text for a curated set of common
# general-knowledge topics — replaces live Wikipedia API fetching, which
# repeatedly failed in deployment (rate limiting, User-Agent rejection,
# inconsistent JSON responses). This removes that entire class of problem:
# no network dependency for this portion at all, so it's exactly as
# reliable as the SQuAD portion, which has been solid throughout.
STATIC_WIKI_ARTICLES = {
    "Artificial intelligence": (
        "Artificial intelligence (AI) is the intelligence of machines or software, as opposed "
        "to the intelligence of humans or animals. It is also the field of study in computer "
        "science that develops and studies intelligent machines. AI technology is widely used "
        "throughout industry, government, and science. Some high-profile applications include "
        "advanced web search engines, recommendation systems, understanding human speech, "
        "self-driving cars, generative and creative tools, and automated decision-making. "
        "AI was founded as an academic discipline in 1956. Approaches to AI include machine "
        "learning, where systems learn patterns from data rather than following explicitly "
        "programmed rules, and symbolic AI, which relies on formal logic and hand-crafted "
        "knowledge representations. Modern AI systems increasingly rely on neural networks, "
        "which are loosely inspired by the structure of the human brain and are trained on "
        "large amounts of data to recognize patterns and make predictions."
    ),
    "Machine learning": (
        "Machine learning is a field of study in artificial intelligence concerned with the "
        "development of statistical algorithms that can learn from data and generalize to "
        "unseen data, without being explicitly programmed to do so. Machine learning approaches "
        "are traditionally divided into three broad categories: supervised learning, where "
        "algorithms are trained on labeled example data; unsupervised learning, where algorithms "
        "find patterns in unlabeled data; and reinforcement learning, where an agent learns to "
        "make decisions by receiving rewards or penalties for its actions. Deep learning, a "
        "subset of machine learning, uses multi-layered neural networks to progressively extract "
        "higher-level features from raw input. Machine learning is used in a wide variety of "
        "applications, including medicine, email filtering, speech recognition, agriculture, "
        "and computer vision, where it is difficult to develop conventional algorithms."
    ),
    "Computer science": (
        "Computer science is the study of computation, information, and automation. Computer "
        "science spans theoretical disciplines, such as algorithms, computation theory, and "
        "information theory, to applied disciplines including the design and implementation of "
        "hardware and software. Computer science, built upon a foundation of theoretical "
        "linguistics, discrete mathematics, and electrical engineering, studies the nature and "
        "limits of computation. Subfields include computability theory, which studies what "
        "problems can be solved by computation, computational complexity theory, which studies "
        "the time and resources required to solve problems, database design, computer "
        "networking, and artificial intelligence. Computer scientists are involved in areas "
        "ranging from the design of individual microcontrollers to the analysis of algorithms "
        "and the study of computer performance."
    ),
    "Natural language processing": (
        "Natural language processing (NLP) is a subfield of computer science and artificial "
        "intelligence concerned with giving computers the ability to process and understand "
        "human language. Major tasks in NLP include text classification, machine translation, "
        "question answering, sentiment analysis, and text summarization. Modern NLP systems are "
        "largely based on machine learning, especially large neural network models called "
        "language models that are trained on large amounts of text data. These models learn "
        "statistical relationships between words and phrases, allowing them to generate "
        "coherent text, answer questions, and perform many other language-related tasks."
    ),
    "Neural network": (
        "An artificial neural network is a computational model inspired by the structure and "
        "function of biological neural networks in animal brains. A neural network consists of "
        "connected units called artificial neurons, organized in layers. Each connection can "
        "transmit a signal to other neurons, and the strength of these connections, called "
        "weights, is adjusted during training. Neural networks are used to approximate functions "
        "that can depend on a large number of inputs and are generally unknown. Deep neural "
        "networks, which have many layers, are the foundation of modern deep learning and are "
        "used in applications such as image recognition, speech recognition, and natural "
        "language processing."
    ),
    "Photosynthesis": (
        "Photosynthesis is a biological process used by plants, algae, and some bacteria to "
        "convert light energy into chemical energy, through a process that converts carbon "
        "dioxide and water into glucose and oxygen. Photosynthesis is vital for life on Earth "
        "because it produces the oxygen in the atmosphere and forms the base of most food chains. "
        "The process occurs primarily in the chloroplasts of plant cells, using a green pigment "
        "called chlorophyll to absorb light. Photosynthesis consists of two main stages: the "
        "light-dependent reactions, which capture energy from sunlight, and the light-independent "
        "reactions, also called the Calvin cycle, which use that energy to build glucose from "
        "carbon dioxide."
    ),
    "Prime number": (
        "A prime number is a natural number greater than 1 that has no positive divisors other "
        "than 1 and itself. A natural number greater than 1 that is not prime is called a "
        "composite number. For example, 5 is prime because the only ways of writing it as a "
        "product, 1 x 5 or 5 x 1, involve 5 itself. However, 6 is composite because it is the "
        "product of two numbers, 2 x 3, that are both smaller than 6. Prime numbers are central "
        "in number theory because of the fundamental theorem of arithmetic, which states that "
        "every natural number greater than 1 can be represented uniquely as a product of prime "
        "numbers. Prime numbers are also important in modern cryptography, particularly in "
        "public-key encryption systems."
    ),
    "Climate change": (
        "Climate change refers to long-term shifts in temperatures and weather patterns. These "
        "shifts may be natural, but since the 1800s, human activities have been the main driver "
        "of climate change, primarily due to the burning of fossil fuels like coal, oil, and gas, "
        "which produces heat-trapping greenhouse gases. Consequences of climate change include "
        "more intense droughts, water scarcity, severe fires, rising sea levels, flooding, "
        "melting polar ice, and declining biodiversity. Reducing greenhouse gas emissions "
        "through renewable energy adoption, energy efficiency, and changes in land use is "
        "considered essential to limiting further warming."
    ),
    "World War II": (
        "World War II was a global conflict that lasted from 1939 to 1945, involving the vast "
        "majority of the world's countries, including all of the great powers, organized into "
        "two opposing military alliances: the Allies and the Axis. It was the deadliest conflict "
        "in human history, resulting in an estimated 70 to 85 million fatalities. The war began "
        "with Germany's invasion of Poland in September 1939 and ended in 1945 with the surrender "
        "of Japan following the atomic bombings of Hiroshima and Nagasaki. The war led to "
        "significant political changes, including the emergence of the United States and the "
        "Soviet Union as rival superpowers, setting the stage for the Cold War."
    ),
    "Solar System": (
        "The Solar System is the gravitationally bound system of the Sun and the objects that "
        "orbit it, including eight planets, their moons, dwarf planets, asteroids, and comets. "
        "The four inner, terrestrial planets are Mercury, Venus, Earth, and Mars, composed "
        "primarily of rock and metal. The four outer, giant planets are Jupiter, Saturn, Uranus, "
        "and Neptune, composed largely of hydrogen, helium, and other gases and ices. The Solar "
        "System formed about 4.6 billion years ago from the gravitational collapse of a giant "
        "interstellar molecular cloud. The Sun contains more than 99 percent of the Solar "
        "System's total mass."
    ),
    "DNA": (
        "Deoxyribonucleic acid, or DNA, is the molecule that carries genetic instructions for "
        "the development, functioning, growth, and reproduction of all known living organisms. "
        "DNA is composed of two long strands forming a double helix, made up of units called "
        "nucleotides. Each nucleotide contains one of four nitrogenous bases: adenine, thymine, "
        "guanine, and cytosine. The sequence of these bases encodes genetic information, similar "
        "to how a sequence of letters spells out words. DNA is passed from parents to offspring "
        "during reproduction, and it is the basis for heredity, determining traits such as eye "
        "color and susceptibility to certain diseases."
    ),
    "Democracy": (
        "Democracy is a system of government in which political power is vested in the people, "
        "who exercise that power directly or through elected representatives. Key features of "
        "democracy typically include free and fair elections, protection of civil liberties, "
        "rule of law, and separation of powers among different branches of government. Democracy "
        "originated in ancient Greece, particularly in Athens, where citizens voted directly on "
        "legislation. Modern democracies are generally representative, where citizens elect "
        "officials to make decisions on their behalf, rather than voting on every issue directly."
    ),
    "Evolution": (
        "Evolution is the change in heritable characteristics of biological populations over "
        "successive generations. The theory of evolution by natural selection was first "
        "articulated by Charles Darwin in his 1859 book On the Origin of Species. Natural "
        "selection occurs when individuals with certain heritable traits are more likely to "
        "survive and reproduce than others, causing those traits to become more common in a "
        "population over time. Evidence for evolution comes from multiple independent fields, "
        "including fossil records, comparative anatomy, and genetics, which show that all life "
        "on Earth shares common ancestry."
    ),
    "Internet": (
        "The Internet is a global system of interconnected computer networks that uses the "
        "Internet protocol suite to communicate between networks and devices. It is a network "
        "of networks that consists of private, public, academic, business, and government "
        "networks, linked by electronic, wireless, and optical networking technologies. The "
        "Internet carries a vast range of information resources and services, such as the "
        "World Wide Web, electronic mail, telephony, and file sharing. The origins of the "
        "Internet date back to research commissioned in the 1960s to build robust, fault-"
        "tolerant communication networks."
    ),
    "Renewable energy": (
        "Renewable energy is energy derived from naturally replenishing resources that are "
        "not depleted when used, such as sunlight, wind, rain, tides, waves, and geothermal "
        "heat. Common renewable energy technologies include solar photovoltaic panels, wind "
        "turbines, hydroelectric dams, and geothermal power plants. Renewable energy is "
        "considered essential for reducing greenhouse gas emissions and combating climate "
        "change, since it produces little to no direct emissions during operation, unlike "
        "fossil fuels. Costs of renewable technologies, particularly solar and wind, have "
        "fallen dramatically over the past decade, making them increasingly competitive with "
        "traditional energy sources."
    ),
    "Quantum mechanics": (
        "Quantum mechanics is a fundamental theory in physics that describes the behavior of "
        "nature at the scale of atoms and subatomic particles. Unlike classical physics, "
        "quantum mechanics allows particles to exist in multiple states simultaneously, a "
        "property known as superposition, and to become correlated in ways that classical "
        "physics cannot explain, known as entanglement. Quantum mechanics underlies many modern "
        "technologies, including semiconductors, lasers, and magnetic resonance imaging. It also "
        "forms the basis for emerging technologies such as quantum computing, which uses "
        "quantum-mechanical phenomena to perform computations that would be infeasible for "
        "classical computers."
    ),
    "Human brain": (
        "The human brain is the central organ of the human nervous system, responsible for "
        "processing sensory information, controlling movement, and enabling complex cognitive "
        "functions such as thought, memory, and language. The brain contains approximately 86 "
        "billion neurons, which communicate with each other through electrical and chemical "
        "signals across connections called synapses. The brain is divided into several major "
        "regions, including the cerebrum, cerebellum, and brainstem, each responsible for "
        "different functions. The cerebral cortex, the outer layer of the cerebrum, is "
        "particularly associated with higher-order functions such as reasoning and "
        "consciousness."
    ),
    "Economics": (
        "Economics is the social science that studies the production, distribution, and "
        "consumption of goods and services. Economics is broadly divided into microeconomics, "
        "which examines the behavior of individuals and firms in making decisions regarding "
        "the allocation of scarce resources, and macroeconomics, which studies economy-wide "
        "phenomena such as inflation, unemployment, and economic growth. Key economic concepts "
        "include supply and demand, opportunity cost, and market equilibrium. Economic systems "
        "vary widely across the world, ranging from market economies, where resource allocation "
        "is primarily determined by supply and demand, to planned economies, where a central "
        "authority makes production decisions."
    ),
    "William Shakespeare": (
        "William Shakespeare was an English playwright, poet, and actor, widely regarded as the "
        "greatest writer in the English language. His extant works consist of some 39 plays, "
        "154 sonnets, and a few other verses. His plays are divided into tragedies, comedies, "
        "and histories, and include well-known works such as Hamlet, Romeo and Juliet, Macbeth, "
        "and A Midsummer Night's Dream. Shakespeare's plays have been translated into every "
        "major living language and are performed more often than those of any other playwright. "
        "He was born in Stratford-upon-Avon in 1564 and died in 1616."
    ),
    "Ancient Rome": (
        "Ancient Rome was a civilization that grew from a small city on the Italian Peninsula "
        "into a vast empire that spanned much of Europe, North Africa, and the Middle East. "
        "Roman civilization is often grouped into three main periods: the Roman Kingdom, the "
        "Roman Republic, and the Roman Empire. At its height, the Roman Empire controlled "
        "approximately 5 million square kilometers of land and governed tens of millions of "
        "people. Rome is credited with major contributions to law, government, engineering, "
        "and architecture, including aqueducts, roads, and monumental buildings such as the "
        "Colosseum. The Western Roman Empire fell in 476 CE, though the Eastern Roman Empire, "
        "known as the Byzantine Empire, continued for nearly another thousand years."
    ),
    "Mount Everest": (
        "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur "
        "Himal sub-range of the Himalayas, on the border between Nepal and the Tibet Autonomous "
        "Region of China. Its elevation is 8,849 meters. Mount Everest attracts climbers from "
        "around the world, including experienced mountaineers as well as, controversially, "
        "novice climbers willing to pay large sums to professional guides. The mountain poses "
        "serious risks, including altitude sickness, weather, and high winds, and hundreds of "
        "climbers have died attempting to reach the summit. The first confirmed ascent was made "
        "by Edmund Hillary and Tenzing Norgay in 1953."
    ),
    "Python (programming language)": (
        "Python is a high-level, general-purpose programming language known for its emphasis "
        "on code readability, using significant indentation. Python is dynamically typed and "
        "garbage-collected, and it supports multiple programming paradigms, including "
        "structured, object-oriented, and functional programming. Python is widely used in "
        "web development, data analysis, artificial intelligence, scientific computing, and "
        "automation, largely due to its extensive standard library and the availability of "
        "third-party packages. Python was created by Guido van Rossum and first released in "
        "1991. It consistently ranks among the most popular programming languages in the world."
    ),
    "Gravity": (
        "Gravity, or gravitational force, is a fundamental interaction that causes mutual "
        "attraction between all things that have mass or energy. On Earth, gravity gives "
        "weight to physical objects and causes the ocean tides. Gravity is responsible for "
        "many of the observed astronomical phenomena, including the orbits of planets around "
        "the Sun and the structure of galaxies. Isaac Newton's law of universal gravitation, "
        "published in 1687, describes gravity as a force acting between two masses. In the "
        "20th century, Albert Einstein's general theory of relativity provided a more accurate "
        "description of gravity as the curvature of spacetime caused by mass and energy."
    ),
}

_model = None

# One shared ChromaDB client for the whole app — both this file (building
# the knowledge base) and retrieval_agent.py (querying it) use this exact
# same connection. Previously each created its own separate client, which
# could fail to see the other's writes reliably (two connections to the
# same on-disk store, opened at different times). Sharing one connection
# avoids that entirely.
_chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)


def get_embedding_model():
    """Loads the embedding model once and reuses it across the app,
    since it's also used by retrieval_agent.py at query time."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def _get_static_wiki_articles() -> list[str]:
    """No network calls — returns the hardcoded article text directly.
    Guaranteed to work every time, regardless of Render's network
    conditions or Wikipedia's API behavior."""
    print(f"Loading {len(STATIC_WIKI_ARTICLES)} static Wikipedia-derived articles (no network calls)...")
    return list(STATIC_WIKI_ARTICLES.values())


def build_knowledge_base(force_rebuild: bool = False):
    """
    Builds (or reuses) the ChromaDB knowledge base. If the collection
    already has data and force_rebuild is False, this is a no-op — this
    matters locally, where you don't want to re-download and re-embed
    everything on every restart, but on Render, the collection is always
    empty on a fresh container, so this runs fresh every time there.
    """
    if force_rebuild and os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    client = _chroma_client
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    if collection.count() > 0:
        print(f"Knowledge base already populated with {collection.count()} chunks — skipping rebuild.")
        return collection

    print("Building knowledge base...")

    print(f"Loading SQuAD (first {SQUAD_ROW_LIMIT} rows)...")
    # streaming=True here too — non-streaming loads were caching far more
    # data to disk than needed for a small slice, which caused a separate
    # "temporary storage exceeded" failure during deploy
    squad_stream = load_dataset("rajpurkar/squad", split="train", streaming=True)

    squad_rows = []
    for i, row in enumerate(itertools.islice(squad_stream, SQUAD_ROW_LIMIT), start=1):
        squad_rows.append(row)
        if i % 5000 == 0 or i == SQUAD_ROW_LIMIT:
            print(f"  SQuAD: {i}/{SQUAD_ROW_LIMIT} rows loaded")

    squad_contexts = list(set(row["context"] for row in squad_rows))

    wiki_texts = _get_static_wiki_articles()

    all_documents = squad_contexts + wiki_texts

    all_chunks = []
    for doc in all_documents:
        all_chunks.extend(_chunk_text(doc))

    print(f"Embedding {len(all_chunks)} chunks...")
    model = get_embedding_model()

    # Manual batching with explicit progress prints — sentence-transformers'
    # built-in tqdm progress bar often doesn't render cleanly in Render's
    # log viewer (it relies on carriage-return updates, which get lost in
    # a piped/non-interactive log stream), making a long embedding step
    # look stuck even when it's progressing normally.
    EMBED_BATCH_SIZE = 64  # raised from 16 now that Standard tier has headroom
    embeddings_list = []
    for i in range(0, len(all_chunks), EMBED_BATCH_SIZE):
        batch = all_chunks[i:i + EMBED_BATCH_SIZE]
        batch_embeddings = model.encode(batch, batch_size=EMBED_BATCH_SIZE)
        embeddings_list.extend(batch_embeddings)
        done = min(i + EMBED_BATCH_SIZE, len(all_chunks))
        if done % 1000 < EMBED_BATCH_SIZE or done == len(all_chunks):
            print(f"  Embedded {done}/{len(all_chunks)} chunks")

    embeddings = np.array(embeddings_list)

    BATCH_SIZE = 5000
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch_chunks = all_chunks[i:i + BATCH_SIZE]
        batch_embeddings = embeddings[i:i + BATCH_SIZE]
        collection.add(
            ids=[str(j) for j in range(i, i + len(batch_chunks))],
            embeddings=batch_embeddings.tolist(),
            documents=batch_chunks,
        )
        print(f"  Indexed {i + len(batch_chunks)}/{len(all_chunks)} chunks")

    print(
        f"Knowledge base built: {len(squad_contexts)} SQuAD contexts, "
        f"{len(wiki_texts)} Wikipedia articles, {len(all_chunks)} total chunks."
    )
    return collection


if __name__ == "__main__":
    build_knowledge_base(force_rebuild=True)
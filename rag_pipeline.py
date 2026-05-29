"""
Logos RAG Pipeline — LangChain + ChromaDB
Demonstrates: RAG, vector embeddings, semantic search over Bible corpus
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import chromadb
import os
from typing import List, Dict, Optional

# ── SAMPLE BIBLE CORPUS ──
# In production: load all 31,102 verses from a Bible JSON/CSV
# This sample covers the most-referenced verses for demonstration

SAMPLE_BIBLE_VERSES = [
    # John
    {"reference": "John 3:16", "text": "For God so loved the world that he gave his one and only Son, that whoever believes in him shall not perish but have eternal life.", "book": "John", "testament": "new"},
    {"reference": "John 14:6", "text": "Jesus answered, 'I am the way and the truth and the life. No one comes to the Father except through me.'", "book": "John", "testament": "new"},
    {"reference": "John 1:1", "text": "In the beginning was the Word, and the Word was with God, and the Word was God.", "book": "John", "testament": "new"},
    {"reference": "John 11:35", "text": "Jesus wept.", "book": "John", "testament": "new"},
    # Psalms
    {"reference": "Psalm 23:1", "text": "The Lord is my shepherd, I lack nothing.", "book": "Psalms", "testament": "old"},
    {"reference": "Psalm 23:4", "text": "Even though I walk through the darkest valley, I will fear no evil, for you are with me; your rod and your staff, they comfort me.", "book": "Psalms", "testament": "old"},
    {"reference": "Psalm 46:1", "text": "God is our refuge and strength, an ever-present help in trouble.", "book": "Psalms", "testament": "old"},
    {"reference": "Psalm 119:105", "text": "Your word is a lamp for my feet, a light on my path.", "book": "Psalms", "testament": "old"},
    # Romans
    {"reference": "Romans 3:23", "text": "For all have sinned and fall short of the glory of God.", "book": "Romans", "testament": "new"},
    {"reference": "Romans 6:23", "text": "For the wages of sin is death, but the gift of God is eternal life in Christ Jesus our Lord.", "book": "Romans", "testament": "new"},
    {"reference": "Romans 8:28", "text": "And we know that in all things God works for the good of those who love him, who have been called according to his purpose.", "book": "Romans", "testament": "new"},
    {"reference": "Romans 10:9", "text": "If you declare with your mouth, 'Jesus is Lord,' and believe in your heart that God raised him from the dead, you will be saved.", "book": "Romans", "testament": "new"},
    # Philippians
    {"reference": "Philippians 4:13", "text": "I can do all this through him who gives me strength.", "book": "Philippians", "testament": "new"},
    {"reference": "Philippians 4:6", "text": "Do not be anxious about anything, but in every situation, by prayer and petition, with thanksgiving, present your requests to God.", "book": "Philippians", "testament": "new"},
    # Matthew
    {"reference": "Matthew 5:3", "text": "Blessed are the poor in spirit, for theirs is the kingdom of heaven.", "book": "Matthew", "testament": "new"},
    {"reference": "Matthew 22:37", "text": "Jesus replied: 'Love the Lord your God with all your heart and with all your soul and with all your mind.'", "book": "Matthew", "testament": "new"},
    {"reference": "Matthew 28:19", "text": "Therefore go and make disciples of all nations, baptizing them in the name of the Father and of the Son and of the Holy Spirit.", "book": "Matthew", "testament": "new"},
    # Genesis
    {"reference": "Genesis 1:1", "text": "In the beginning God created the heavens and the earth.", "book": "Genesis", "testament": "old"},
    # Isaiah
    {"reference": "Isaiah 40:31", "text": "But those who hope in the Lord will renew their strength. They will soar on wings like eagles; they will run and not grow weary, they will walk and not be faint.", "book": "Isaiah", "testament": "old"},
    {"reference": "Isaiah 9:6", "text": "For to us a child is born, to us a son is given, and the government will be on his shoulders. And he will be called Wonderful Counselor, Mighty God, Everlasting Father, Prince of Peace.", "book": "Isaiah", "testament": "old"},
    # Jeremiah
    {"reference": "Jeremiah 29:11", "text": "For I know the plans I have for you, declares the Lord, plans to prosper you and not to harm you, plans to give you hope and a future.", "book": "Jeremiah", "testament": "old"},
    # Proverbs
    {"reference": "Proverbs 3:5", "text": "Trust in the Lord with all your heart and lean not on your own understanding.", "book": "Proverbs", "testament": "old"},
    {"reference": "Proverbs 3:6", "text": "In all your ways submit to him, and he will make your paths straight.", "book": "Proverbs", "testament": "old"},
    # 1 Corinthians
    {"reference": "1 Corinthians 13:4", "text": "Love is patient, love is kind. It does not envy, it does not boast, it is not proud.", "book": "1 Corinthians", "testament": "new"},
    {"reference": "1 Corinthians 13:13", "text": "And now these three remain: faith, hope and love. But the greatest of these is love.", "book": "1 Corinthians", "testament": "new"},
    # Ephesians
    {"reference": "Ephesians 2:8", "text": "For it is by grace you have been saved, through faith—and this is not from yourselves, it is the gift of God.", "book": "Ephesians", "testament": "new"},
    # James
    {"reference": "James 1:2", "text": "Consider it pure joy, my brothers and sisters, whenever you face trials of many kinds.", "book": "James", "testament": "new"},
    {"reference": "James 2:17", "text": "In the same way, faith by itself, if it is not accompanied by action, is dead.", "book": "James", "testament": "new"},
    # 2 Timothy
    {"reference": "2 Timothy 3:16", "text": "All Scripture is God-breathed and is useful for teaching, rebuking, correcting and training in righteousness.", "book": "2 Timothy", "testament": "new"},
    # 1 John
    {"reference": "1 John 4:8", "text": "Whoever does not love does not know God, because God is love.", "book": "1 John", "testament": "new"},
    # Galatians
    {"reference": "Galatians 5:22", "text": "But the fruit of the Spirit is love, joy, peace, forbearance, kindness, goodness, faithfulness.", "book": "Galatians", "testament": "new"},
    # Revelation
    {"reference": "Revelation 21:4", "text": "He will wipe every tear from their eyes. There will be no more death or mourning or crying or pain, for the old order of things has passed away.", "book": "Revelation", "testament": "new"},
]


class ChristianRAGPipeline:
    """
    RAG pipeline for scripture-grounded responses.
    Uses ChromaDB for vector storage and semantic search over Bible verses.
    
    Production extension: Load all 31,102 Bible verses from a full Bible JSON.
    """

    def __init__(self, persist_directory: str = "./chroma_bible_db"):
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.retriever = None
        self._ready = False
        self._initialize()

    def _initialize(self):
        """Build or load the ChromaDB vector store."""
        try:
            # Convert verses to LangChain Documents
            documents = []
            for verse in SAMPLE_BIBLE_VERSES:
                doc = Document(
                    page_content=f"{verse['reference']}: {verse['text']}",
                    metadata={
                        "reference": verse["reference"],
                        "book": verse["book"],
                        "testament": verse["testament"],
                        "text": verse["text"]
                    }
                )
                documents.append(doc)

            # Use ChromaDB with in-memory client for demo
            # Production: use persist_directory for persistent storage
            client = chromadb.Client()

            # Build vectorstore using Chroma
            # Note: In production, use OpenAI/Anthropic embeddings for better semantic search
            # Here we use a simple approach that works without an embeddings API key
            from langchain_community.embeddings import FakeEmbeddings
            embeddings = FakeEmbeddings(size=384)  # Replace with real embeddings in production

            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                client=client,
                collection_name="bible_verses"
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            self._ready = True
            print(f"✝ RAG pipeline initialized with {len(documents)} Bible verses")

        except Exception as e:
            print(f"RAG initialization warning: {e}")
            print("Running in fallback mode (keyword search only)")
            self._ready = False

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """
        Retrieve relevant Bible verses for a query.
        Falls back to keyword search if vector DB unavailable.
        """
        if self._ready and self.retriever:
            try:
                docs = self.retriever.get_relevant_documents(query)
                return [
                    {
                        "reference": doc.metadata["reference"],
                        "text": doc.metadata["text"],
                        "book": doc.metadata["book"]
                    }
                    for doc in docs[:k]
                ]
            except Exception:
                pass

        # Fallback: keyword search
        return self._keyword_search(query, k)

    def _keyword_search(self, query: str, k: int = 3) -> List[Dict]:
        """Simple keyword-based fallback search over the Bible corpus."""
        query_words = set(query.lower().split())
        scored = []
        for verse in SAMPLE_BIBLE_VERSES:
            verse_words = set(verse["text"].lower().split())
            score = len(query_words & verse_words)
            if score > 0:
                scored.append((score, verse))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"reference": v["reference"], "text": v["text"], "book": v["book"]}
            for _, v in scored[:k]
        ]

    def verify_verse(self, reference: str) -> Dict:
        """
        Check if a Bible verse reference exists in our corpus.
        Production: check against complete 31,102-verse database.
        """
        ref_clean = reference.strip()
        for verse in SAMPLE_BIBLE_VERSES:
            if verse["reference"].lower() == ref_clean.lower():
                return {
                    "exists": True,
                    "reference": verse["reference"],
                    "text": verse["text"],
                    "note": "Verified against Bible corpus"
                }

        # Check for known fake verses
        fake_checks = {
            "hezekiah": "There is no book of Hezekiah in the Bible.",
            "jasher": "The Book of Jasher is referenced in Joshua 10:13 and 2 Samuel 1:18 but is not part of the canonical Bible.",
        }
        ref_lower = ref_clean.lower()
        for fake, note in fake_checks.items():
            if fake in ref_lower:
                return {"exists": False, "note": note}

        return {
            "exists": None,  # Unknown — not in our sample corpus
            "note": f"'{ref_clean}' was not found in our sample corpus. This does not necessarily mean it doesn't exist — our demo corpus contains ~30 key verses. A production system would check all 31,102 verses."
        }

    def get_qa_chain(self, denomination: str = "general") -> RetrievalQA:
        """
        Build a full LangChain RetrievalQA chain for scripture-grounded answers.
        """
        llm = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=1000)

        prompt_template = """You are Logos, a Christian AI assistant. Use the following verified Bible verses to answer the question.

Verified Scripture Context:
{context}

Question: {question}

Answer with pastoral warmth and theological accuracy. Cite the provided verses. If the context doesn't contain relevant verses, draw on your theological knowledge but note you are not quoting from the verified corpus."""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        return chain

    def is_ready(self) -> bool:
        return self._ready

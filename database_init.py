# database_init.py
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
from backend import Base, SQLALCHEMY_DATABASE_URL

def init_database():
    # Create engine
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    
    # Create database if it doesn't exist
    if not database_exists(engine.url):
        create_database(engine.url)
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    print("Database and tables created successfully!")

if __name__ == "__main__":
    init_database()
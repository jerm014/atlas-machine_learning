import random
from collections import deque
from typing import List, Optional, Deque

class SkipBoCard:
    def __init__(self, actual_value: int, is_wild: bool = False):
        if not (1 <= actual_value <= 12):
            raise ValueError("Card value must be between 1 and 12")
            
        self.actual_value = actual_value
        self.is_wild = is_wild
        self.played_value = None
        
    def play_as(self, value: int) -> None:
        if not (1 <= value <= 12):
            raise ValueError("Played value must be between 1 and 12")
            
        if not self.is_wild and value != self.actual_value:
            raise ValueError("Non-wild cards must be played as their actual value")
            
        self.played_value = value
        
    def reset(self) -> None:
        self.played_value = None
        
    def __str__(self) -> str:
        status = "Wild" if self.is_wild else f"Value: {self.actual_value}"
        played = f", Played as: {self.played_value}" if self.played_value else ""
        return f"SkipBo Card ({status}{played})"

class Deck:
    def __init__(self):
        self.cards: Deque[SkipBoCard] = deque()
        
    def shuffle(self) -> None:
        cards_list = list(self.cards)
        random.shuffle(cards_list)
        self.cards = deque(cards_list)
        
    def add_card(self, card: SkipBoCard) -> None:
        self.cards.append(card)
        
    def draw_card(self) -> Optional[SkipBoCard]:
        return self.cards.pop() if self.cards else None
        
    def size(self) -> int:
        return len(self.cards)
    
    def is_empty(self) -> bool:
        return len(self.cards) == 0

class DrawPile(Deck):
    @classmethod
    def create_initial_deck(cls) -> 'DrawPile':
        deck = cls()
        # Add regular cards (12 sets of 1-12)
        for _ in range(12):
            for value in range(1, 13):
                deck.add_card(SkipBoCard(value))
        # Add wild cards (18 total)
        for _ in range(18):
            deck.add_card(SkipBoCard(1, is_wild=True))
        deck.shuffle()
        return deck

class StockPile(Deck):
    def peek_top(self) -> Optional[SkipBoCard]:
        return self.cards[-1] if self.cards else None

class DiscardPile(Deck):
    def peek_top(self) -> Optional[SkipBoCard]:
        return self.cards[-1] if self.cards else None

class PlayedPile(Deck):
    pass

class BuildPile(Deck):
    def __init__(self):
        super().__init__()
        self.current_value = 0
        
    def can_play_card(self, card: SkipBoCard) -> bool:
        next_value = self.current_value + 1
        return (card.is_wild or card.actual_value == next_value) and next_value <= 12
        
    def play_card(self, card: SkipBoCard) -> bool:
        if not self.can_play_card(card):
            return False
            
        self.current_value += 1
        card.play_as(self.current_value)
        self.add_card(card)
        
        if self.current_value == 12:
            self.reset_pile()
            
        return True
        
    def reset_pile(self) -> List[SkipBoCard]:
        cards_to_played = list(self.cards)
        self.cards.clear()
        self.current_value = 0
        return cards_to_played

class SkipBoGame:
    def __init__(self, num_players: int):
        if not (2 <= num_players <= 6):
            raise ValueError("Number of players must be between 2 and 6")
            
        # Create initial deck and shuffle
        self.draw_pile = DrawPile.create_initial_deck()
        
        # Create player stock piles
        self.stock_piles = [StockPile() for _ in range(num_players)]
        
        # Deal cards to stock piles (30 cards each)
        for pile in self.stock_piles:
            for _ in range(30):
                card = self.draw_pile.draw_card()
                if card:
                    pile.add_card(card)
                    
        # Create discard piles (4 per player)
        self.discard_piles = [[DiscardPile() for _ in range(4)] for _ in range(num_players)]
        
        # Create build piles (4 shared piles)
        self.build_piles = [BuildPile() for _ in range(4)]
        
        # Create played pile
        self.played_pile = PlayedPile()
        
    def move_completed_build_to_played(self, build_pile_index: int) -> None:
        if build_pile_index < 0 or build_pile_index >= len(self.build_piles):
            raise ValueError("Invalid build pile index")
            
        completed_cards = self.build_piles[build_pile_index].reset_pile()
        for card in completed_cards:
            self.played_pile.add_card(card)

# Start a new game with 3 players
game = SkipBoGame(3)

# Check a player's stock pile
top_card = game.stock_piles[0].peek_top()
print(f"Player 1's top stock card: {top_card}")

# Draw a card from the draw pile
card = game.draw_pile.draw_card()
print(f"Drew card: {card}")

# Play a card on a build pile
if game.build_piles[0].can_play_card(card):
    game.build_piles[0].play_card(card)
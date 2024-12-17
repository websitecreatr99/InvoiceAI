from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional

# Define the Pydantic models for the schema
class Address(BaseModel):
    name: Optional[str]
    address1: Optional[str] = None
    address2: Optional[str] = None
    address3: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    country_code: Optional[str] = None
    zip: Optional[str] = None

class Contact(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    contact_number: Optional[str] = None

class Header(BaseModel):
    ship_to: Optional[Address] = None
    bill_to: Optional[Address] = None
    vendor: Optional[Address] = None
    buyer_contact: Optional[Contact] = None
    shipping_contact: Optional[Contact] = None
    project_number: Optional[str] = None
    purchase_order_number: Optional[str] = None
    job_name: Optional[str] = None
    job_number: Optional[str] = None
    quote_number: Optional[str] = None
    date_ordered: Optional[str] = None
    delivery_date: Optional[str] = None
    shipping_instructions: Optional[str] = None
    notes: Optional[str] = None
    ship_via: Optional[str] = None
    payment_terms: Optional[str] = None

class LineItem(BaseModel):
    line_no: Optional[str] = None
    on_hand: Optional[str] = None
    to_buy: Optional[str] = None
    quantity: Optional[int] = None
    uom: Optional[str] = None
    unit_price: Optional[str] = None
    currency: Optional[str] = None
    part_numbers: Optional[str] = None
    product_description: Optional[str] = None
    spell_corrected_product_description: Optional[str] = None

class ExtractionItem(BaseModel):
    header: Optional[Header] = None
    line_items: Optional[List[LineItem]] = None

class Extraction(BaseModel):
    extraction: List[ExtractionItem]
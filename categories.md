---
layout: page
title: Categories
permalink: /categories/
---

<div class="categories-page">
  {% for category in site.categories %}
    <div class="category-section">
      <h2 id="{{ category[0] | slugify }}">{{ category[0] | capitalize }}</h2>
      <ul>
        {% for post in category[1] %}
          <li>
            <a href="{{ post.url }}">{{ post.title }}</a>
            <span class="post-date">({{ post.date | date: "%B %-d, %Y" }})</span>
          </li>
        {% endfor %}
      </ul>
    </div>
  {% endfor %}
</div> 